"""Model 2 - Speaker VAE.
Improvements: classifier over TRAINING speakers only, stronger speaker loss (1.0), smaller KL with warmup,
30% of inputs pass through the frozen AE (robust to cleaned microphone references), validation accuracy,
best-checkpoint saving, and export of per-segment and per-speaker-mean embeddings for Models 3-4."""
import random
import torch
import torch.nn.functional as F
from .common import (device, paths, load_cache, load_meta, iterate, n_batches, mel_batch, warmup_cosine,
                     make_scaler, optim_step, fit, safe_save, seed_everything, tload, LATENT_DIM)
from .models import SpeakerVAE
from .train_ae import load_ae, augment, corrupt


def kl_term(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def load_vae(root):
    meta = load_meta(paths(root))
    vae = SpeakerVAE(LATENT_DIM, num_speakers=len(meta["train_speakers"])).to(device)
    vae.load_state_dict(tload(f"{paths(root)['ckpt']}/vae_best.pt", map_location=device))
    vae.eval()
    for q in vae.parameters():
        q.requires_grad_(False)
    return vae


def train(root, epochs=50, batch_size=32, lr=1e-3, warmup=500, kl_max=5e-4, kl_warmup_epochs=10,
          speaker_weight=1.0, p_through_ae=0.3, seed=0):
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    meta = load_meta(p)
    spk2idx = {s: i for i, s in enumerate(meta["train_speakers"])}
    labels = torch.tensor([spk2idx.get(s, -1) for s in C["spks"]])
    tr_idx, va_idx = C["idx"]["train"], C["idx"]["val"]
    ae = load_ae(root)

    model = SpeakerVAE(LATENT_DIM, num_speakers=len(spk2idx)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = n_batches(len(tr_idx), batch_size)
    sched = warmup_cosine(opt, warmup, epochs * steps)
    scaler = make_scaler(False)

    def train_one_epoch(epoch):
        model.train()
        kl_w = kl_max * min(1.0, (epoch + 1) / kl_warmup_epochs)
        sums, n = {"train": 0.0, "recon": 0.0, "kl": 0.0, "spk_ce": 0.0}, 0
        for idx in iterate(tr_idx, batch_size):
            clean = augment(mel_batch(C, idx))
            y = labels[idx].to(device)
            x = clean
            if random.random() < p_through_ae:
                with torch.no_grad():
                    x = ae(corrupt(clean))
            recon, mu, logvar, logits = model(x)
            r, k, c = F.mse_loss(recon, clean), kl_term(mu, logvar), F.cross_entropy(logits, y)
            loss = r + kl_w * k + speaker_weight * c
            optim_step(loss, opt, scaler, model.parameters(), clip=1.0)
            sched.step()
            for key, v in zip(sums, (loss, r, k, c)):
                sums[key] += v.item()
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate():
        model.eval()
        r_sum, c_sum, correct, total, n = 0.0, 0.0, 0, 0, 0
        for idx in iterate(va_idx, 128, shuffle=False, drop_last=False):
            mel = mel_batch(C, idx)
            y = labels[idx].to(device)
            mu, _ = model.encode(mel)
            logits = model.speaker_classifier(mu)
            r_sum += F.mse_loss(model.decode(mu), mel).item()
            c_sum += F.cross_entropy(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item(); total += len(idx); n += 1
        n = max(n, 1)
        return {"val": r_sum / n + c_sum / n, "val_spk_acc": correct / max(total, 1)}

    hist = fit("vae", p["ckpt"], epochs, train_one_epoch, validate,
               best_weights=lambda: model.state_dict(),
               modules={"model": model}, optimizers={"opt": opt}, schedulers={"sched": sched})
    export_embeddings(root)
    return hist


@torch.no_grad()
def export_embeddings(root, batch_size=256):
    """Saves mu for every segment (N,64) and the mean embedding per speaker (train speakers: train+val
    segments; holdout speakers: their own segments, used only for zero-shot tests)."""
    p = paths(root)
    C = load_cache(p)
    vae = load_vae(root)
    N = C["mels"].size(0)
    mus = []
    for idx in iterate(torch.arange(N), batch_size, shuffle=False, drop_last=False):
        mus.append(vae.encode(mel_batch(C, idx))[0].cpu())
    mus = torch.cat(mus)
    spks = C["spks"]
    speaker_mean = {}
    for s in sorted(set(spks)):
        rows = [i for i, x in enumerate(spks) if x == s]
        speaker_mean[s] = mus[rows].mean(0)
    safe_save(mus, p["vae_mu"])
    safe_save(speaker_mean, p["speaker_mean"])
    print(f"Exported {N} segment embeddings and {len(speaker_mean)} speaker means.")
