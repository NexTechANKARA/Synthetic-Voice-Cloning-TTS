"""Model 3 (upgraded) - FastSpeech-style Transformer TTS.

Why: the baseline Transformer predicted 128 fixed frames from learned slots, with no information about
how long each sound lasts, so it learned an "average phrase" and could not place sounds in time.

How durations are learned (no external aligner, no new data prep):
  * an alignment module scores every (mel frame, character) pair;
  * a forward-sum (CTC-style) loss teaches it which characters can produce which frames, in order;
  * monotonic alignment search (Glow-TTS) turns the soft scores into whole frames per character;
  * those frame counts train the decoder (via the length regulator) and the duration predictor.
At inference the duration predictor decides the timing, so output length follows the text.
"""
import math
import random
import torch
import torch.nn.functional as F
from .common import (device, paths, load_cache, load_meta, iterate, n_batches, warmup_cosine, make_scaler,
                     optim_step, fit, safe_save, tload, seed_everything, SEG)
from .models import FastSpeechTTS
from .train_transformer import SpeakerSampler


# ---------------------------------------------------------------- alignment helpers
def mel_lengths(mels):
    """Real (non-padding) frame count of each cached mel. Padding frames are exactly 0. mels: (N, 80, T)."""
    active = (mels.float().amax(dim=1) > 0)                       # (N, T)
    T = active.size(1)
    last = T - torch.flip(active, dims=[1]).float().argmax(dim=1)  # index after the last active frame
    return torch.where(active.any(dim=1), last, torch.zeros_like(last))


def beta_binomial_log_prior(text_lens, mel_lens, L, T, scaling=1.0):
    """Log prior (B, T, L) that favours a diagonal alignment early in training (RAD-TTS)."""
    B = text_lens.size(0)
    dev = text_lens.device
    k = torch.arange(L, device=dev, dtype=torch.float32)[None, None, :]          # character index
    t = torch.arange(1, T + 1, device=dev, dtype=torch.float32)[None, :, None]   # frame index (1-based)
    n = (text_lens.float() - 1).clamp(min=0)[:, None, None]
    Tm = mel_lens.float()[:, None, None]
    a = scaling * t
    b = scaling * (Tm - t + 1).clamp(min=1e-3)
    lchoose = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k).clamp(min=0) + 1)
    lbeta = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
    logp = lchoose + lbeta(k + a, (n - k).clamp(min=0) + b) - lbeta(a, b)
    valid = (k <= n) & (t <= Tm)
    return torch.where(valid, logp, torch.zeros_like(logp)).expand(B, T, L)


@torch.no_grad()
def monotonic_alignment_search(logp, text_lens, mel_lens):
    """Best monotonic path through logp (B, T, L): starts at (0, 0), ends at (T_b-1, L_b-1), each step
    stays on the same character or moves to the next one. Returns integer durations (B, L)."""
    B, T, L = logp.shape
    neg = torch.finfo(logp.dtype).min / 4
    tmask = torch.arange(L, device=logp.device)[None] < text_lens[:, None]
    lp = logp.masked_fill(~tmask[:, None, :], neg)
    value = torch.full((B, T, L), neg, device=logp.device, dtype=logp.dtype)
    moved = torch.zeros((B, T, L), dtype=torch.bool, device=logp.device)
    value[:, 0, 0] = lp[:, 0, 0]
    for t in range(1, T):
        prev = value[:, t - 1]
        shifted = torch.cat([torch.full_like(prev[:, :1], neg), prev[:, :-1]], dim=1)
        moved[:, t] = shifted > prev
        value[:, t] = lp[:, t] + torch.maximum(prev, shifted)
    durations = torch.zeros((B, L), dtype=torch.long, device=logp.device)
    idx = (text_lens - 1).clamp(min=0).clone()
    ar = torch.arange(B, device=logp.device)
    for t in range(T - 1, -1, -1):                     # no boolean indexing: avoids a GPU sync per step
        active = (t < mel_lens).long()
        durations.scatter_add_(1, idx[:, None], active[:, None])
        step = active * moved[ar, t, idx].long() * (idx > 0).long()
        idx = idx - step
    return durations


def forward_sum_loss(attn_logprob, text_lens, mel_lens, blank_logprob=-1.0):
    """CTC-style loss: every character must be used, in order (RAD-TTS). attn_logprob: (B, T, L)."""
    B, T, L = attn_logprob.shape
    x = F.pad(attn_logprob, (1, 0), value=blank_logprob)                        # (B, T, L+1), 0 = blank
    cls = torch.arange(L + 1, device=x.device)[None, None]
    x = x.masked_fill(cls > text_lens[:, None, None], -1e4)
    x = torch.log_softmax(x, dim=-1).transpose(0, 1)                             # (T, B, L+1)
    targets = torch.arange(1, L + 1, device=x.device)[None].expand(B, L)
    return F.ctc_loss(x, targets, mel_lens, text_lens, blank=0, reduction="mean", zero_infinity=True)


# ---------------------------------------------------------------- data
def usable_rows(C, rows, mlens, tlens):
    """Keep segments with some audio and no more characters than frames (MAS needs one frame per char)."""
    keep = (mlens[rows] >= 4) & (tlens[rows] >= 1) & (tlens[rows] <= mlens[rows])
    return rows[keep]


class Batcher:
    def __init__(self, C, sampler):
        self.C, self.sampler = C, sampler
        self.mlens = mel_lengths(C["mels"])
        self.tlens = C["text_mask"].sum(1).long()

    def __call__(self, idx, randomize=True):
        C = self.C
        mel = C["mels"][idx].float().to(device)                 # (B, 80, 128)
        ids = C["text_ids"][idx].to(device)
        mask = C["text_mask"][idx].to(device)
        tl, ml = self.tlens[idx].to(device), self.mlens[idx].to(device)
        L, T = int(tl.max()), int(ml.max())
        return dict(mel=mel[:, :, :T], ids=ids[:, :L], mask=mask[:, :L], tlen=tl, mlen=ml,
                    spk=self.sampler(idx, randomize=randomize))


def run_model(model, b, use_prior=True):
    """One forward pass with durations from the alignment. Returns losses and the alignment."""
    h = model.encode(b["ids"], b["mask"], b["spk"])
    B, L = b["ids"].shape
    T = b["mel"].size(2)
    prior = beta_binomial_log_prior(b["tlen"], b["mlen"], L, T) if use_prior else None
    logp = model.align(b["ids"], b["mask"], b["mel"], prior)      # (B, T, L)
    frame_valid = torch.arange(T, device=device)[None] < b["mlen"][:, None]
    logp = logp.masked_fill(~frame_valid[:, :, None], 0.0)
    fs = forward_sum_loss(logp, b["tlen"], b["mlen"])
    dur = monotonic_alignment_search(logp.detach(), b["tlen"], b["mlen"])   # (B, L), sums to mel length

    pred, frame_mask = model.decode(h, dur, b["spk"], max_len=T)
    m = frame_mask.unsqueeze(1).float()
    denom = m.sum() * pred.size(1)
    l1 = ((pred - b["mel"]).abs() * m).sum() / denom
    l2 = (((pred - b["mel"]) ** 2) * m).sum() / denom

    logd = model.duration(h.detach(), b["mask"])
    target = torch.log1p(dur.float())
    dl = (((logd - target) ** 2) * b["mask"]).sum() / b["mask"].sum()
    return dict(l1=l1, l2=l2, dur=dl, fs=fs), dur, pred


def load_fastspeech(root):
    meta = load_meta(paths(root))
    m = FastSpeechTTS(meta["vocab_size"]).to(device)
    m.load_state_dict(tload(f"{paths(root)['ckpt']}/fastspeech_best.pt", map_location=device))
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def init_from_baseline(model, root):
    """Copy the baseline Transformer's encoder weights (text embedding, encoder, speaker projection)."""
    import os
    path = f"{paths(root)['ckpt']}/transformer_best.pt"
    if not os.path.exists(path):
        print("No baseline checkpoint found; encoder starts from scratch.")
        return
    old = tload(path, map_location=device)
    own = model.state_dict()
    shared = {k: v for k, v in old.items()
              if k.startswith(("text_embed.", "text_encoder.", "speaker_proj.")) and k in own and own[k].shape == v.shape}
    model.load_state_dict(shared, strict=False)
    print(f"Encoder initialised from the baseline Transformer ({len(shared)} tensors).")


# ---------------------------------------------------------------- training
def train(root, epochs=60, stop_after=None, batch_size=32, lr=3e-4, warmup=4000, p_mean=0.5,
          w_fs=1.0, w_dur=1.0, prior_epochs=10, seed=0, warm_start=False):
    """epochs = length of the LR schedule. stop_after = pause after this many epochs (run again to continue)."""
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    meta = load_meta(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]), p_mean)
    batcher = Batcher(C, sampler)
    tr_idx = usable_rows(C, C["idx"]["train"], batcher.mlens, batcher.tlens)
    va_idx = usable_rows(C, C["idx"]["val"], batcher.mlens, batcher.tlens)
    print(f"Training segments: {len(tr_idx)} (skipped {len(C['idx']['train']) - len(tr_idx)} with more "
          f"characters than frames) | validation: {len(va_idx)}")

    model = FastSpeechTTS(meta["vocab_size"]).to(device)
    import os
    if warm_start and not os.path.exists(f"{p['ckpt']}/fastspeech_resume.pt"):
        init_from_baseline(model, root)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
    steps = n_batches(len(tr_idx), batch_size)
    sched = warmup_cosine(opt, warmup, epochs * steps)
    scaler = make_scaler(False)

    def train_one_epoch(epoch):
        model.train()
        use_prior = epoch < prior_epochs
        s, n = {"train": 0.0, "mel": 0.0, "dur": 0.0, "align": 0.0}, 0
        for idx in iterate(tr_idx, batch_size):
            b = batcher(idx)
            ls, _, _ = run_model(model, b, use_prior)
            loss = ls["l1"] + ls["l2"] + w_dur * ls["dur"] + w_fs * ls["fs"]
            if not torch.isfinite(loss):
                continue
            optim_step(loss, opt, scaler, model.parameters(), clip=1.0)
            sched.step()
            s["train"] += loss.item(); s["mel"] += ls["l1"].item()
            s["dur"] += ls["dur"].item(); s["align"] += ls["fs"].item(); n += 1
        return {k: v / max(n, 1) for k, v in s.items()}

    @torch.no_grad()
    def validate():
        model.eval()
        tot, dsum, n = 0.0, 0.0, 0
        for idx in iterate(va_idx, 64, shuffle=False, drop_last=False):
            b = batcher(idx, randomize=False)
            ls, _, _ = run_model(model, b, use_prior=False)
            tot += ls["l1"].item(); dsum += ls["dur"].item(); n += 1
        return {"val": tot / max(n, 1), "val_dur": dsum / max(n, 1)}

    stop = min(stop_after, epochs) if stop_after else epochs
    hist = fit("fastspeech", p["ckpt"], stop, train_one_epoch, validate,
               best_weights=lambda: model.state_dict(),
               modules={"model": model}, optimizers={"opt": opt}, schedulers={"sched": sched})
    if stop >= epochs:
        export_coarse(root, p_mean=p_mean)
    return hist


@torch.no_grad()
def alignment_report(root, n=3):
    """For the early check: alignment images and how sharp/diagonal they are, on validation segments."""
    p = paths(root)
    C = load_cache(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]))
    batcher = Batcher(C, sampler)
    va = usable_rows(C, C["idx"]["val"], batcher.mlens, batcher.tlens)[:n]
    model = load_fastspeech(root)
    b = batcher(va, randomize=False)
    h = model.encode(b["ids"], b["mask"], b["spk"])
    logp = model.align(b["ids"], b["mask"], b["mel"])
    out = []
    for j in range(len(va)):
        T, L = int(b["mlen"][j]), int(b["tlen"][j])
        A = logp[j, :T, :L].exp().cpu()
        out.append({"text": C["texts"][int(va[j])], "attn": A, "sharpness": float(A.max(1).values.mean())})
    return out


@torch.no_grad()
def export_coarse(root, p_mean=0.5, batch_size=64, seed=0):
    """Coarse mels for Model 4, made with durations from the alignment so they line up frame by frame
    with the real mels. Same file format as before: (N, 80, 128) float16 + the speaker embeddings."""
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]), p_mean)
    batcher = Batcher(C, sampler)
    model = load_fastspeech(root)
    split = C["split"]
    N = C["mels"].size(0)
    coarse = torch.zeros(N, C["mels"].size(1), SEG, dtype=torch.float16)
    embs = torch.zeros(N, 64)
    ok = (batcher.mlens >= 4) & (batcher.tlens >= 1) & (batcher.tlens <= batcher.mlens)
    for idx in iterate(torch.arange(N)[ok], batch_size, shuffle=False, drop_last=False):
        train_rows = [split[i] == "train" for i in idx.tolist()]
        b = batcher(idx, randomize=True)
        e_mean = sampler(idx, randomize=False)
        b["spk"] = torch.where(torch.tensor(train_rows, device=device)[:, None], b["spk"], e_mean)
        h = model.encode(b["ids"], b["mask"], b["spk"])
        logp = model.align(b["ids"], b["mask"], b["mel"])
        T = b["mel"].size(2)
        logp = logp.masked_fill(~(torch.arange(T, device=device)[None] < b["mlen"][:, None])[:, :, None], 0.0)
        dur = monotonic_alignment_search(logp, b["tlen"], b["mlen"])
        mel, _ = model.decode(h, dur, b["spk"], max_len=T)
        coarse[idx, :, :T] = mel.half().cpu()
        embs[idx] = b["spk"].cpu()
    safe_save({"coarse": coarse, "spk_emb": embs}, p["coarse"])
    print(f"Exported coarse mels: {N} (aligned with the real mels)")


# ---------------------------------------------------------------- diffusion data with predicted durations
def warp_to_durations(real, d_true, d_pred, max_len=SEG):
    """Re-time a real mel (80, T) from its true per-character durations to the predicted ones.
    Each character's real frames are resampled (nearest neighbour) to its predicted frame count, so the
    result keeps the recording's spectral detail but follows the predicted timing. Returns (80, max_len)."""
    d_true, d_pred = d_true.long(), d_pred.long().clamp(min=0)
    out = real.new_zeros(real.size(0), max_len)
    if int(d_pred.sum()) == 0:
        return out
    start_true = torch.cumsum(d_true, 0) - d_true      # computed on ALL characters, so positions stay correct
                                                       # even when a character gets 0 predicted frames
    start_pred = torch.cumsum(d_pred, 0) - d_pred
    char = torch.repeat_interleave(torch.arange(len(d_pred), device=real.device), d_pred)[:max_len]
    pos = torch.arange(len(char), device=real.device) - start_pred[char]
    src = start_true[char] + ((pos.float() + 0.5) * d_true[char].float() / d_pred[char].float()).long()
    src = torch.minimum(src, start_true[char] + (d_true[char] - 1).clamp(min=0)).clamp(0, real.size(1) - 1)
    out[:, :len(char)] = real[:, src]
    return out


@torch.no_grad()
def export_coarse_predicted(root, p_mean=0.5, batch_size=64, seed=0):
    """Coarse mels made with PREDICTED durations (exactly what the pipeline produces at inference), paired with
    the real mels warped to the same timing. Saved to data/coarse_pred_cache.pt for diffusion training."""
    seed_everything(seed)
    p = paths(root)
    C = load_cache(p)
    sampler = SpeakerSampler(C, tload(p["vae_mu"]), tload(p["speaker_mean"]), p_mean)
    batcher = Batcher(C, sampler)
    model = load_fastspeech(root)
    split = C["split"]
    N, M = C["mels"].size(0), C["mels"].size(1)
    coarse = torch.zeros(N, M, SEG, dtype=torch.float16)
    target = torch.zeros(N, M, SEG, dtype=torch.float16)
    embs = torch.zeros(N, 64)
    ok = (batcher.mlens >= 4) & (batcher.tlens >= 1) & (batcher.tlens <= batcher.mlens)
    from tqdm.auto import tqdm
    rows = torch.arange(N)[ok]
    for idx in tqdm(list(iterate(rows, batch_size, shuffle=False, drop_last=False)), desc="exporting"):
        train_rows = [split[i] == "train" for i in idx.tolist()]
        b = batcher(idx, randomize=True)
        e_mean = sampler(idx, randomize=False)
        b["spk"] = torch.where(torch.tensor(train_rows, device=device)[:, None], b["spk"], e_mean)
        h = model.encode(b["ids"], b["mask"], b["spk"])
        T = b["mel"].size(2)
        logp = model.align(b["ids"], b["mask"], b["mel"])
        logp = logp.masked_fill(~(torch.arange(T, device=device)[None] < b["mlen"][:, None])[:, :, None], 0.0)
        d_true = monotonic_alignment_search(logp, b["tlen"], b["mlen"])
        d_pred = model.predict_durations(h, b["mask"])
        mel, _ = model.decode(h, d_pred, b["spk"], max_len=SEG)
        coarse[idx, :, :mel.size(2)] = mel.half().cpu()
        for j in range(len(idx)):
            L = int(b["tlen"][j])
            target[idx[j]] = warp_to_durations(b["mel"][j], d_true[j, :L], d_pred[j, :L]).half().cpu()
        embs[idx] = b["spk"].cpu()
    safe_save({"coarse": coarse, "target": target, "spk_emb": embs}, p["coarse_pred"])
    print(f"Exported {int(ok.sum())} coarse mels with predicted durations + warped real targets")
