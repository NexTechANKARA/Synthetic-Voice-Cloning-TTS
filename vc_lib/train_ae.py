"""Model 1 - Denoising Autoencoder.
Improvements: aligned segments from 36 speakers, random noise strength per sample, gain + time-shift
augmentation, L1+MSE loss, AdamW + warmup/cosine LR, validation split, best-checkpoint saving."""
import torch
import torch.nn.functional as F
from .common import (device, paths, load_cache, iterate, n_batches, mel_batch, warmup_cosine, make_scaler, tload,
                     optim_step, fit, seed_everything)
from .models import DenoisingAutoencoder


def augment(mel, max_gain=0.05, max_shift=16):
    """Random gain on speech frames only (silence stays silent) + random right shift padded with silence."""
    B = mel.size(0)
    g = (torch.rand(B, 1, 1, 1, device=mel.device) * 2 - 1) * max_gain
    mel = torch.where(mel > 0, (mel + g).clamp(0, 1), mel)
    s = int(torch.randint(0, max_shift + 1, (1,)))
    if s:
        mel = torch.cat([torch.zeros_like(mel[..., :s]), mel[..., :-s]], dim=-1)
    return mel


def corrupt(clean, noise_min=0.03, noise_max=0.3, generator=None):
    B = clean.size(0)
    std = torch.empty(B, 1, 1, 1).uniform_(noise_min, noise_max, generator=generator).to(clean.device)
    noise = torch.randn(clean.shape, generator=generator).to(clean.device)
    return (clean + noise * std).clamp(0, 1)


def load_ae(root):
    ae = DenoisingAutoencoder().to(device)
    ae.load_state_dict(tload(f"{paths(root)['ckpt']}/ae_best.pt", map_location=device))
    ae.eval()
    for q in ae.parameters():
        q.requires_grad_(False)
    return ae


def train(root, epochs=50, batch_size=32, lr=1e-3, warmup=500, noise_min=0.03, noise_max=0.3, seed=0):
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    tr_idx, va_idx = C["idx"]["train"], C["idx"]["val"]
    model = DenoisingAutoencoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = n_batches(len(tr_idx), batch_size)
    sched = warmup_cosine(opt, warmup, epochs * steps)
    scaler = make_scaler(False)

    def train_one_epoch(epoch):
        model.train()
        tot, n = 0.0, 0
        for idx in iterate(tr_idx, batch_size):
            clean = augment(mel_batch(C, idx))
            recon = model(corrupt(clean, noise_min, noise_max))
            loss = F.l1_loss(recon, clean) + F.mse_loss(recon, clean)
            optim_step(loss, opt, scaler, model.parameters(), clip=1.0)
            sched.step()
            tot += loss.item(); n += 1
        return {"train": tot / max(n, 1)}

    @torch.no_grad()
    def validate():
        model.eval()
        g = torch.Generator().manual_seed(123)          # same noise every epoch -> comparable numbers
        tot, n = 0.0, 0
        for idx in iterate(va_idx, 128, shuffle=False, drop_last=False):
            clean = mel_batch(C, idx)
            recon = model(corrupt(clean, noise_min, noise_max, generator=g))
            tot += (F.l1_loss(recon, clean) + F.mse_loss(recon, clean)).item(); n += 1
        return {"val": tot / max(n, 1)}

    return fit("ae", p["ckpt"], epochs, train_one_epoch, validate,
               best_weights=lambda: model.state_dict(),
               modules={"model": model}, optimizers={"opt": opt}, schedulers={"sched": sched})
