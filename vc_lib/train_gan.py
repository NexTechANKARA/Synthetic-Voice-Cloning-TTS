"""Model 5 - GAN vocoder.
Trains on random 1.49 s crops of full TRAINING utterances (about 58,000 crops from 76 voices).
Validation uses crops from the 4 UNSEEN speakers, so the score reflects new voices.
Only crops are loaded (not per-segment audio), which keeps RAM within the standard Colab limit.
Final epochs use aligned degraded real mels so the vocoder copes with imperfect model mels."""
import random
import torch
import torch.nn.functional as F
import torchaudio
from .common import (device, paths, iterate, fit, tload, from_int16, seed_everything, EMA,
                     SR, N_FFT, HOP, N_MELS)
from .models import Generator, MultiScaleDiscriminator


def degrade(mel):
    """mel (B,1,80,128) in [0,1]: mimic Transformer blur and diffusion noise, keeping time alignment."""
    k = random.choice([1, 3, 5])
    if k > 1:
        mel = F.avg_pool2d(mel, (k, k), stride=1, padding=k // 2, count_include_pad=False)
    return (mel + 0.02 * torch.randn_like(mel)).clamp(0, 1)


def load_generator(root):
    G = Generator().to(device)
    G.load_state_dict(tload(f"{paths(root)['ckpt']}/gan_best.pt", map_location=device))
    G.eval()
    for q in G.parameters():
        q.requires_grad_(False)
    return G


def train(root, epochs=100, steps_per_epoch=1000, batch_size=16, lr=2e-4, lr_decay=0.999,
          w_mel=45.0, w_fm=2.0, ema_decay=0.999, degrade_last_epochs=30, p_degrade=0.5, seed=0):
    seed_everything(seed)
    p = paths(root)
    tr = tload(p["gan_crops"])
    ho = tload(p["gan_holdout"])
    mels, wavs = tr["mels"], tr["wavs"]
    N = mels.size(0)
    assert N > 0, "No training crops - rebuild the cache (notebook 00)"
    n_ev = min(64, ho["mels"].size(0))
    ev_mels, ev_wavs = ho["mels"][:n_ev].clone(), ho["wavs"][:n_ev].clone()
    del tr, ho
    print(f"Vocoder training crops: {N} | unseen-speaker validation crops: {n_ev}")

    mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
                                                  n_mels=N_MELS, power=1.0).to(device)

    def log_mel(w):
        return torch.log(torch.clamp(mel_tf(w.squeeze(1)), min=1e-5))

    def sample_batch(bs):
        k = torch.randint(0, N, (bs,))
        mel = mels[k].float().unsqueeze(1).to(device)
        real = from_int16(wavs[k]).unsqueeze(1).to(device)
        return mel, real

    G, D = Generator().to(device), MultiScaleDiscriminator().to(device)
    ema = EMA(G, ema_decay)
    opt_g = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99))
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, lr_decay)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, lr_decay)

    def train_one_epoch(epoch):
        G.train(); D.train()
        use_degrade = epoch >= epochs - degrade_last_epochs
        s = {"g": 0.0, "d": 0.0, "mel": 0.0}
        for _ in range(steps_per_epoch):
            mel, real = sample_batch(batch_size)
            if use_degrade and random.random() < p_degrade:
                mel = degrade(mel)
            with torch.no_grad():
                fake = G(mel)
            d_real, _ = D(real)
            d_fake, _ = D(fake)
            d_loss = sum(F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f))
                         for r, f in zip(d_real, d_fake))
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()

            fake = G(mel)
            d_fake, feats_fake = D(fake)
            with torch.no_grad():
                _, feats_real = D(real)
            adv = sum(F.mse_loss(f, torch.ones_like(f)) for f in d_fake)
            fm = sum(F.l1_loss(a, b) for fa, fb in zip(feats_fake, feats_real) for a, b in zip(fa, fb))
            mel_l = F.l1_loss(log_mel(fake), log_mel(real))
            g_loss = adv + w_fm * fm + w_mel * mel_l
            opt_g.zero_grad(set_to_none=True); g_loss.backward(); opt_g.step()
            ema.update(G)
            s["g"] += g_loss.item(); s["d"] += d_loss.item(); s["mel"] += mel_l.item()
        sch_g.step(); sch_d.step()
        return {k: v / steps_per_epoch for k, v in s.items()}

    @torch.no_grad()
    def validate():
        tot, n = 0.0, 0
        for idx in iterate(torch.arange(n_ev), 16, shuffle=False, drop_last=False):
            mel = ev_mels[idx].float().unsqueeze(1).to(device)
            real = from_int16(ev_wavs[idx]).unsqueeze(1).to(device)
            tot += F.l1_loss(log_mel(ema.shadow(mel)), log_mel(real)).item(); n += 1
        return {"val": tot / max(n, 1)}

    return fit("gan", p["ckpt"], epochs, train_one_epoch, validate,
               best_weights=lambda: ema.shadow.state_dict(),
               modules={"G": G, "D": D}, optimizers={"g": opt_g, "d": opt_d},
               schedulers={"g": sch_g, "d": sch_d}, emas={"G": ema})
