"""Model 4 - Diffusion refiner.
Improvements: cosine noise schedule, condition noise so training coarse mels look like inference ones,
classifier-free guidance (condition dropped to SILENCE = -1, not 0), EMA weights, DDIM 50-step sampling,
mixed precision, validation on fixed noise + best (EMA) checkpoint."""
import math
import torch
import torch.nn.functional as F
from .common import (device, paths, load_cache, iterate, n_batches, warmup_cosine, make_scaler, autocast,
                     optim_step, fit, tload, seed_everything, EMA)
from .models import DiffusionDenoiser

NULL_COND = -1.0          # "no condition" = a silent mel in diffusion space


def to_diff(x01):
    return x01 * 2 - 1


class Schedule:
    def __init__(self, T=200, kind="cosine"):
        if kind == "cosine":
            s = 0.008
            t = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            ab = f[1:] / f[0]
            prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
            beta = (1 - ab / prev).clamp(1e-8, 0.999)
        else:
            beta = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
        self.T = T
        self.beta = beta.float().to(device)
        self.alpha_bar = torch.cumprod(1 - beta, 0).float().to(device)

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


@torch.no_grad()
def ddim_sample(model, sched, coarse01, spk, steps=50, cfg=1.5, seed=None, strength=1.0):
    """coarse01: (B,1,80,T) in [0,1]. Deterministic DDIM (eta=0) with classifier-free guidance.
    strength: 1.0 = full diffusion from pure noise (original behaviour).
              < 1.0 = shallow diffusion (DiffSinger): start from the coarse mel noised to step
              strength*T and denoise only from there, so words and timing stay those of the coarse mel
              and diffusion only refines detail. 0 returns the coarse mel unchanged."""
    model.eval()
    if strength <= 0:
        return coarse01.clamp(0, 1)
    B = coarse01.size(0)
    cond = to_diff(coarse01)
    null = torch.full_like(cond, NULL_COND)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    noise = torch.randn(cond.shape, generator=g).to(device)
    t_start = max(1, min(sched.T - 1, int(round(strength * (sched.T - 1)))))
    n_steps = max(2, int(round(steps * t_start / (sched.T - 1))))
    ts = torch.linspace(t_start, 0, n_steps).long().tolist()
    if strength >= 1.0:
        x = noise
    else:
        x = sched.q_sample(cond, torch.full((B,), t_start, device=device, dtype=torch.long), noise)
    for i, t in enumerate(ts):
        ab = sched.alpha_bar[t]
        ab_prev = sched.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        tt = torch.full((2 * B,), t, device=device, dtype=torch.long)
        eps_c, eps_u = model(torch.cat([x, x]), tt, torch.cat([cond, null]), torch.cat([spk, spk])).float().chunk(2)
        eps = eps_u + cfg * (eps_c - eps_u)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()          # keep eps consistent with clamped x0
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return ((x + 1) / 2).clamp(0, 1)


def load_diffusion(root, name="diffusion"):
    m = DiffusionDenoiser().to(device)
    m.load_state_dict(tload(f"{paths(root)['ckpt']}/{name}_best.pt", map_location=device))
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def train(root, epochs=60, batch_size=32, lr=2e-4, warmup=1000, T=200, p_uncond=0.1, cond_noise=0.03,
          ema_decay=0.999, amp=True, seed=0, name="diffusion", coarse_file=None):
    """name: checkpoint name. coarse_file: which coarse cache to train on (default: the aligned one).
    If the cache holds a "target" tensor (real mels time-warped to the predicted durations), the model learns
    to map those coarse mels to it; otherwise the targets are the real mels from the data cache."""
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    CC = tload(coarse_file or p["coarse"])
    coarse, spk_emb = CC["coarse"], CC["spk_emb"]
    targets = CC.get("target", C["mels"])
    print(f"[{name}] training on {'warped targets (predicted durations)' if 'target' in CC else 'real mels'}")
    tr_idx, va_idx = C["idx"]["train"], C["idx"]["val"]
    sched_d = Schedule(T)

    model = DiffusionDenoiser().to(device)
    ema = EMA(model, ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = n_batches(len(tr_idx), batch_size)
    sched = warmup_cosine(opt, warmup, epochs * steps, min_ratio=0.1)
    scaler = make_scaler(amp)

    def batch(idx):
        x0 = to_diff(targets[idx].float().unsqueeze(1)).to(device)
        c = coarse[idx].float().unsqueeze(1).to(device)
        return x0, c, spk_emb[idx].to(device)

    def train_one_epoch(epoch):
        model.train()
        tot, n = 0.0, 0
        for idx in iterate(tr_idx, batch_size):
            x0, c01, spk = batch(idx)
            cond = to_diff((c01 + cond_noise * torch.randn_like(c01)).clamp(0, 1))
            drop = torch.rand(cond.size(0), device=device) < p_uncond
            cond = torch.where(drop[:, None, None, None], torch.full_like(cond, NULL_COND), cond)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            with autocast(amp):
                pred = model(sched_d.q_sample(x0, t, noise), t, cond, spk)
            loss = F.mse_loss(pred.float(), noise)
            optim_step(loss, opt, scaler, model.parameters(), clip=1.0)
            sched.step()
            ema.update(model)
            tot += loss.item(); n += 1
        return {"train": tot / max(n, 1)}

    @torch.no_grad()
    def validate():
        net = ema.shadow
        g = torch.Generator().manual_seed(123)
        tot, n = 0.0, 0
        for idx in iterate(va_idx, 128, shuffle=False, drop_last=False):
            x0, c01, spk = batch(idx)
            t = torch.randint(0, T, (x0.size(0),), generator=g).to(device)
            noise = torch.randn(x0.shape, generator=g).to(device)
            with autocast(amp):
                pred = net(sched_d.q_sample(x0, t, noise), t, to_diff(c01), spk)
            tot += F.mse_loss(pred.float(), noise).item(); n += 1
        return {"val": tot / max(n, 1)}

    return fit(name, p["ckpt"], epochs, train_one_epoch, validate,
               best_weights=lambda: ema.shadow.state_dict(),
               modules={"model": model}, optimizers={"opt": opt}, schedulers={"sched": sched},
               emas={"ema": ema}, scaler=scaler)
