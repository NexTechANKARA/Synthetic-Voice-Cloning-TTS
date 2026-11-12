"""Shared code for every notebook. Every model MUST use these functions so preprocessing never drifts."""
import os, json, math, random, re, time, copy, shutil
import numpy as np
import torch

# ---------------- constants (same as the original project) ----------------
SR, N_MELS, N_FFT, HOP, SEG = 22050, 80, 1024, 256, 128
WAV_LEN = SEG * HOP                  # 32768 samples, ~1.486 s
MAX_TEXT_LEN = 60
PAD_ID, UNK_ID = 0, 1
LATENT_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paths(root):
    p = {"root": root,
         "data": os.path.join(root, "data"),
         "shards": os.path.join(root, "data", "shards"),
         "ckpt": os.path.join(root, "checkpoints")}
    for k in ("data", "shards", "ckpt"):
        os.makedirs(p[k], exist_ok=True)
    p.update({
        "speakers": os.path.join(p["data"], "speakers.json"),
        "meta": os.path.join(p["data"], "meta.json"),
        "cache": os.path.join(p["data"], "cache.pt"),
        "gan_crops": os.path.join(p["data"], "gan_crops.pt"),
        "gan_holdout": os.path.join(p["data"], "gan_holdout.pt"),
        "vae_mu": os.path.join(p["ckpt"], "vae_mu.pt"),
        "speaker_mean": os.path.join(p["ckpt"], "speaker_mean.pt"),
        "coarse": os.path.join(p["data"], "coarse_cache.pt"),
        "coarse_pred": os.path.join(p["data"], "coarse_pred_cache.pt"),
    })
    return p


# ---------------- file helpers ----------------
def tload(path, map_location="cpu"):
    """Load our own files (they contain dicts/lists/strings, not only weights)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                     # very old torch without weights_only
        return torch.load(path, map_location=map_location)


def safe_save(obj, path):
    """Write to a temp file, then rename, so a Drive hiccup never corrupts the last good file."""
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        try:
            os.replace(tmp, path)
        except Exception:
            shutil.move(tmp, path)
        return True
    except Exception as e:
        print(f"WARNING: could not save {os.path.basename(path)} ({e}). Check Google Drive free space!")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        return False


def load_meta(p):
    with open(p["meta"]) as f:
        return json.load(f)


# ---------------- audio ----------------
def norm_wav(y, peak=0.95):
    """Peak-normalise ONCE per utterance (done in data prep and on reference clips). Never per segment."""
    y = np.asarray(y, dtype=np.float32)
    m = float(np.abs(y).max()) if y.size else 0.0
    return y if m < 1e-8 else y / m * peak


def raw_db(y):
    import librosa
    m = librosa.feature.melspectrogram(y=np.asarray(y, dtype=np.float32), sr=SR, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS, power=2.0)
    return 10.0 * np.log10(np.maximum(m, 1e-10))


def wav_to_mel(y, db_range):
    """Absolute dB scale mapped to [0,1] with a range MEASURED from the data (stored in meta.json).
    No normalisation happens in here, so mel level and waveform level always agree."""
    lo, hi = db_range
    return np.clip((raw_db(y) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fit_frames(mel, n=SEG):
    """Crop / pad with SILENCE (0). No masking anywhere: models learn to output silence after speech."""
    T = mel.shape[1]
    return mel[:, :n] if T >= n else np.pad(mel, ((0, 0), (0, n - T)))


def fit_wav(y, n=WAV_LEN):
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def to_int16(y):
    return torch.from_numpy((np.clip(y, -1, 1) * 32767).astype(np.int16))


def from_int16(t):
    return t.float() / 32767.0


# ---------------- text ----------------
def clean_text(s):
    """lower-case, numbers spelled out, only a-z, apostrophe and single spaces."""
    s = str(s).lower()
    try:
        from num2words import num2words
        s = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", s)
    except ImportError:
        s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    return {c: i + 2 for i, c in enumerate(chars)}


def encode_text(text, char2id, max_len=MAX_TEXT_LEN):
    ids = [char2id.get(c, UNK_ID) for c in text][:max_len]
    if not ids:                       # an all-padding row would make attention produce NaN
        ids = [UNK_ID]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.float32)


# ---------------- data access ----------------
def load_cache(p):
    """Returns the cache dict plus index tensors for each split."""
    C = tload(p["cache"])
    split = np.array(C["split"])
    C["idx"] = {s: torch.from_numpy(np.where(split == s)[0]) for s in ("train", "val", "holdout")}
    return C


def segment_audio(p, C, i):
    """Playable audio for segment i. Segment audio is NOT kept in the cache (RAM); it is read from the shard."""
    sh = tload(os.path.join(p["shards"], f"{C['spks'][i]}.pt"))
    return from_int16(sh["seg_wavs"][C["seg_local"][i]]).numpy()


def iterate(indices, batch_size, shuffle=True, drop_last=True):
    idx = indices[torch.randperm(len(indices))] if shuffle else indices
    stop = len(idx) - batch_size + 1 if drop_last else len(idx)
    for b in range(0, max(stop, 0), batch_size):
        yield idx[b:b + batch_size]


def n_batches(n, batch_size, drop_last=True):
    return max(n // batch_size if drop_last else math.ceil(n / batch_size), 1)


def mel_batch(C, idx, key="mels"):
    return C[key][idx].float().unsqueeze(1).to(device, non_blocking=True)   # (B,1,80,128)


# ---------------- training helpers ----------------
class EMA:
    """Exponential moving average of weights. Sampling with EMA weights is a large, free quality gain."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for q in self.shadow.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, q in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(q.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.05):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def make_scaler(enabled):
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def optim_step(loss, optimizer, scaler, params, clip=1.0):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, clip)
    scaler.step(optimizer)
    scaler.update()


def fit(name, ckpt_dir, epochs, train_one_epoch, validate, best_weights,
        modules, optimizers, schedulers=None, emas=None, scaler=None):
    """Generic epoch loop with auto-resume, history, and best-checkpoint saving.
    train_one_epoch(epoch) -> dict of averages; validate() -> dict containing key 'val' (lower is better)."""
    schedulers, emas = schedulers or {}, emas or {}
    resume_path = os.path.join(ckpt_dir, f"{name}_resume.pt")
    best_path = os.path.join(ckpt_dir, f"{name}_best.pt")
    start, history, best = 0, {}, float("inf")
    if os.path.exists(resume_path):
        try:
            st = tload(resume_path, map_location=device)
            for k, m in modules.items(): m.load_state_dict(st["modules"][k])
            for k, o in optimizers.items(): o.load_state_dict(st["optimizers"][k])
            for k, s in schedulers.items(): s.load_state_dict(st["schedulers"][k])
            for k, e in emas.items(): e.shadow.load_state_dict(st["emas"][k])
            if scaler is not None and st.get("scaler"): scaler.load_state_dict(st["scaler"])
            start, history, best = st["epoch"], st["history"], st["best"]
            print(f"[{name}] resumed at epoch {start} (best val so far {best:.4f})")
        except Exception as e:
            print(f"[{name}] could not resume ({e}); starting fresh")
    else:
        print(f"[{name}] starting fresh")

    for epoch in range(start, epochs):
        t0 = time.time()
        tr = train_one_epoch(epoch)
        va = validate()
        for k, v in {**tr, **va}.items():
            history.setdefault(k, []).append(float(v))
        tag = ""
        if math.isfinite(va["val"]) and va["val"] < best:
            best = va["val"]
            safe_save(best_weights(), best_path)
            tag = "  * best"
        elif not math.isfinite(va["val"]):
            tag = "  !! validation is NaN/inf - lower the learning rate"
        safe_save({"epoch": epoch + 1, "history": history, "best": best,
                   "modules": {k: m.state_dict() for k, m in modules.items()},
                   "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                   "schedulers": {k: s.state_dict() for k, s in schedulers.items()},
                   "emas": {k: e.shadow.state_dict() for k, e in emas.items()},
                   "scaler": scaler.state_dict() if scaler is not None else None}, resume_path)
        msg = "  ".join(f"{k}={v:.4f}" for k, v in {**tr, **va}.items())
        print(f"[{name}] epoch {epoch + 1}/{epochs}  {msg}  ({time.time() - t0:.0f}s){tag}")
    return history


def plot_history(history, keys=None, title=""):
    import matplotlib.pyplot as plt
    keys = keys or list(history.keys())
    plt.figure(figsize=(7, 4))
    for k in keys:
        plt.plot(range(1, len(history[k]) + 1), history[k], label=k)
    plt.xlabel("epoch"); plt.title(title); plt.legend(); plt.grid(True); plt.show()
