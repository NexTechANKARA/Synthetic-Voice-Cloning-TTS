"""Model 3 - Transformer (text -> coarse mel).
Improvements: every training pair is a phrase + the exact audio of that phrase (from data prep);
speaker embedding comes from a DIFFERENT segment of the same speaker or the speaker mean (no content
leak); silence-padded targets so the model learns where speech ends; L1+0.5*MSE; AdamW with warmup +
cosine, gradient clipping, mixed precision; validation + best checkpoint."""
import random
import torch
import torch.nn.functional as F
from .common import (device, paths, load_cache, load_meta, iterate, n_batches, mel_batch, warmup_cosine,
                     make_scaler, autocast, optim_step, fit, safe_save, tload, seed_everything)
from .models import TransformerTTS


class SpeakerSampler:
    """For segment i returns either its speaker's mean embedding (prob p_mean) or the embedding of a
    random OTHER training segment of the same speaker."""
    def __init__(self, C, mus, speaker_mean, p_mean=0.5):
        self.spks, self.mus, self.mean, self.p_mean = C["spks"], mus, speaker_mean, p_mean
        self.by_spk = {}
        for i in C["idx"]["train"].tolist():
            self.by_spk.setdefault(self.spks[i], []).append(i)

    def __call__(self, idx, randomize=True):
        out = []
        for i in idx.tolist():
            s = self.spks[i]
            pool = self.by_spk.get(s, [])
            if not randomize or random.random() < self.p_mean or len(pool) < 2:
                out.append(self.mean[s])
            else:
                j = i
                while j == i:
                    j = pool[random.randrange(len(pool))]
                out.append(self.mus[j])
        return torch.stack(out).to(device)


def load_transformer(root):
    meta = load_meta(paths(root))
    m = TransformerTTS(meta["vocab_size"]).to(device)
    m.load_state_dict(tload(f"{paths(root)['ckpt']}/transformer_best.pt", map_location=device))
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def train(root, epochs=100, batch_size=32, lr=5e-4, warmup=2000, p_mean=0.5, amp=True, seed=0):
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    meta = load_meta(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]), p_mean)
    tr_idx, va_idx = C["idx"]["train"], C["idx"]["val"]

    model = TransformerTTS(meta["vocab_size"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
    steps = n_batches(len(tr_idx), batch_size)
    sched = warmup_cosine(opt, warmup, epochs * steps)
    scaler = make_scaler(amp)

    def batch(idx):
        return (C["text_ids"][idx].to(device), C["text_mask"][idx].to(device), mel_batch(C, idx))

    def train_one_epoch(epoch):
        model.train()
        tot, n = 0.0, 0
        for idx in iterate(tr_idx, batch_size):
            ids, mask, target = batch(idx)
            spk = sampler(idx)
            with autocast(amp):
                pred = model(ids, mask, spk)
            pred = pred.float()
            loss = F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
            optim_step(loss, opt, scaler, model.parameters(), clip=1.0)
            sched.step()
            tot += loss.item(); n += 1
        return {"train": tot / max(n, 1)}

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for idx in iterate(va_idx, 128, shuffle=False, drop_last=False):
            ids, mask, target = batch(idx)
            with autocast(amp):
                pred = model(ids, mask, sampler(idx, randomize=False))
            tot += F.l1_loss(pred.float(), target).item(); n += 1
        return {"val": tot / max(n, 1)}

    hist = fit("transformer", p["ckpt"], epochs, train_one_epoch, validate,
               best_weights=lambda: model.state_dict(),
               modules={"model": model}, optimizers={"opt": opt}, schedulers={"sched": sched}, scaler=scaler)
    export_coarse(root, p_mean=p_mean)
    return hist


@torch.no_grad()
def export_coarse(root, p_mean=0.5, batch_size=128, seed=0):
    """Coarse mels for every segment, made the same way inference will make them (randomised speaker
    embedding for train rows, speaker mean otherwise). Saved with the embedding that produced them."""
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]), p_mean)
    model = load_transformer(root)
    split = C["split"]
    N = C["mels"].size(0)
    coarse, embs = [], []
    for idx in iterate(torch.arange(N), batch_size, shuffle=False, drop_last=False):
        train_rows = torch.tensor([split[i] == "train" for i in idx.tolist()])
        e = sampler(idx, randomize=True)
        e_mean = sampler(idx, randomize=False)
        e = torch.where(train_rows.to(device)[:, None], e, e_mean)
        coarse.append(model(C["text_ids"][idx].to(device), C["text_mask"][idx].to(device), e).squeeze(1).half().cpu())
        embs.append(e.cpu())
    safe_save({"coarse": torch.cat(coarse), "spk_emb": torch.cat(embs)}, p["coarse"])
    print("Exported coarse mels:", N)
