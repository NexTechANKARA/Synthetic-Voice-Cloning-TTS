"""Full pipeline: text -> phrases -> FastSpeech-style Transformer -> Diffusion (DDIM) -> GAN -> joined waveform.
(Falls back to the baseline fixed-length Transformer when no FastSpeech checkpoint exists.)
The AE cleans reference clips before the VAE turns them into a speaker embedding."""
import os
import re
import difflib
import numpy as np
import torch
from .common import (device, paths, load_meta, tload, clean_text, encode_text, norm_wav, wav_to_mel,
                     fit_frames, SR, SEG, HOP)
from .train_ae import load_ae
from .train_vae import load_vae
from .train_transformer import load_transformer
from .train_fastspeech import load_fastspeech
from .train_diffusion import load_diffusion, ddim_sample, Schedule
from .train_gan import load_generator


def split_text(text, max_chars):
    """Split at punctuation into phrases, then pack words into chunks of <= max_chars.
    Returns [(chunk_text, pause_after)]."""
    out = []
    for phrase in re.split(r"(?<=[.,!?;:])\s+|\n+", text.strip()):
        words = clean_text(phrase).split()
        if not words:
            continue
        chunks, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                chunks.append(cur); cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        chunks.append(cur)
        pause = phrase.strip()[-1] in ".,!?;:"
        out += [(c, pause and k == len(chunks) - 1) for k, c in enumerate(chunks)]
    return out


def trim_silence(w, top_db=35, keep_ms=60):
    """Trim leading/trailing silence but keep a short margin, so word endings and soft
    final consonants (s, t, f) are not cut off."""
    import librosa
    _, (a, b) = librosa.effects.trim(w, top_db=top_db)
    keep = int(SR * keep_ms / 1000)
    return w[max(0, a - keep // 2): min(len(w), b + keep)]


def crossfade_concat(pieces, xfade):
    out = np.zeros(0, dtype=np.float32)
    for p in pieces:
        if len(out) > xfade and len(p) > xfade and xfade > 0:
            fade = np.linspace(0, 1, xfade, dtype=np.float32)
            out = np.concatenate([out[:-xfade], out[-xfade:] * (1 - fade) + p[:xfade] * fade, p[xfade:]])
        else:
            out = np.concatenate([out, p])
    return out


class VoicePipeline:
    def __init__(self, root, diffusion_steps=50, cfg=1.0, use_diffusion=True, model="auto", strength=0.1,
                 diffusion="auto"):
        """model: "fastspeech", "baseline", or "auto" (FastSpeech if its checkpoint exists).
        strength: how much diffusion may change the FastSpeech mel (0 = none, 1 = full diffusion from noise).
        Low values (shallow diffusion) keep the words from FastSpeech and only sharpen detail."""
        self.p = paths(root)
        self.meta = load_meta(self.p)
        self.ae, self.vae = load_ae(root), load_vae(root)
        if model == "auto":
            model = "fastspeech" if os.path.exists(f"{self.p['ckpt']}/fastspeech_best.pt") else "baseline"
        self.kind = model
        self.tts = load_fastspeech(root) if model == "fastspeech" else load_transformer(root)
        self.G = load_generator(root)
        self.use_diffusion = use_diffusion
        if diffusion == "auto":
            diffusion = "diffusion"      # original diffusion: tied-best WER at strength 0.1 (see notebook 08)
        self.diff_name = diffusion
        self.diff = load_diffusion(root, diffusion) if use_diffusion else None
        self.sched = Schedule(200)
        self.steps, self.cfg, self.strength = diffusion_steps, cfg, strength
        self.speaker_mean = tload(self.p["speaker_mean"])
        shown = strength if self.kind == "fastspeech" else 1.0
        print(f"Text-to-mel model: {self.kind} | diffusion: "
              f"{f'{self.diff_name}, strength {shown}' if use_diffusion else 'off'}")

    def speakers(self):
        return {"train": self.meta["train_speakers"], "holdout (unseen)": self.meta["holdout"]}

    @torch.no_grad()
    def embed_reference(self, wav_paths, denoise=True):
        """Speaker embedding from one or more reference recordings (any sample rate, ~5-10 s total is good)."""
        import librosa
        windows = []
        for path in ([wav_paths] if isinstance(wav_paths, str) else wav_paths):
            y, _ = librosa.load(path, sr=SR, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)
            mel = wav_to_mel(norm_wav(y), self.meta["db_range"])
            for st in range(0, max(1, mel.shape[1] - SEG + 1), SEG // 2):
                windows.append(fit_frames(mel[:, st:st + SEG]))
        x = torch.from_numpy(np.stack(windows)).unsqueeze(1).to(device)
        if denoise:
            x = self.ae(x)
        return self.vae.encode(x)[0].mean(0)

    @torch.no_grad()
    def text_to_mel(self, text, speaker=None, speaker_emb=None, chunk_chars=None, pace=1.0, seed=0):
        """Returns [(mel (80, T) numpy, pause_after)] per phrase, before the vocoder."""
        if speaker_emb is None:
            speaker_emb = self.speaker_mean[speaker or self.meta["train_speakers"][0]]
        speaker_emb = speaker_emb.to(device)
        max_chars = chunk_chars or (50 if self.kind == "fastspeech" else self.meta["chunk_chars"])
        chunks = split_text(text, max_chars)
        if not chunks:
            return []
        enc = [encode_text(c, self.meta["char2id"]) for c, _ in chunks]
        ids = torch.from_numpy(np.stack([e[0] for e in enc])).to(device)
        mask = torch.from_numpy(np.stack([e[1] for e in enc])).to(device)
        spk = speaker_emb.unsqueeze(0).expand(len(chunks), -1).contiguous()
        if self.kind == "fastspeech":
            coarse, lengths = self.tts.infer(ids, mask, spk, pace=pace)
            T = coarse.size(-1)
            pad = (-T) % 4                                     # the diffusion U-Net halves time twice
            if pad:
                coarse = torch.nn.functional.pad(coarse, (0, pad))
        else:
            coarse = self.tts(ids, mask, spk)
            lengths = torch.full((len(chunks),), coarse.size(-1), device=device)
        strength = self.strength if self.kind == "fastspeech" else 1.0
        mel = ddim_sample(self.diff, self.sched, coarse, spk, self.steps, self.cfg, seed=seed, strength=strength) \
            if self.use_diffusion else coarse
        out = []
        for j, (_, has_pause) in enumerate(chunks):
            n = int(lengths[j])
            out.append((mel[j, 0, :, :n].float().cpu().numpy(), has_pause))
        return out

    @torch.no_grad()
    def synthesize(self, text, speaker=None, speaker_emb=None, chunk_chars=None, pace=1.0, pause_ms=150,
                   xfade_ms=25, trim_db=35, seed=0, return_mels=False):
        """pace < 1 speaks faster, > 1 slower (FastSpeech only)."""
        import librosa
        mels = self.text_to_mel(text, speaker, speaker_emb, chunk_chars, pace, seed)
        if not mels:
            return np.zeros(0, dtype=np.float32)
        pieces, pause = [], np.zeros(int(SR * pause_ms / 1000), dtype=np.float32)
        tail = 12                                     # silent frames so the vocoder can finish the last sound
        for m, has_pause in mels:
            m = np.pad(m, ((0, 0), (2, tail)))
            x = torch.from_numpy(m)[None, None].to(device)
            w = self.G(x).squeeze().cpu().numpy()
            t = trim_silence(w, trim_db)
            pieces.append(t if len(t) > SR // 20 else w)
            if has_pause:
                pieces.append(pause)
        out = norm_wav(crossfade_concat(pieces, int(SR * xfade_ms / 1000)))
        return (out, mels) if return_mels else out


# ---------------- evaluation ----------------
def word_error_rate(ref, hyp):
    r, h = clean_text(ref).split(), clean_text(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def evaluate(pipe, sentences, speaker=None, speaker_emb=None, whisper_size="small"):
    """Intelligibility (Whisper WER) and voice match (cosine of VAE embeddings, output vs target)."""
    import librosa
    from faster_whisper import WhisperModel
    wh = WhisperModel(whisper_size, device="cuda" if device.type == "cuda" else "cpu",
                      compute_type="float16" if device.type == "cuda" else "int8")
    target = (speaker_emb if speaker_emb is not None else pipe.speaker_mean[speaker]).to(device)
    wers, sims = [], []
    for s in sentences:
        y = pipe.synthesize(s, speaker=speaker, speaker_emb=speaker_emb)
        segs, _ = wh.transcribe(librosa.resample(y, orig_sr=SR, target_sr=16000), language="en")
        hyp = " ".join(x.text for x in segs)
        wers.append(word_error_rate(s, hyp))
        mel = wav_to_mel(y, pipe.meta["db_range"])
        wins = [fit_frames(mel[:, st:st + SEG]) for st in range(0, max(1, mel.shape[1] - SEG + 1), SEG // 2)]
        with torch.no_grad():
            e = pipe.vae.encode(torch.from_numpy(np.stack(wins)).unsqueeze(1).to(device))[0].mean(0)
        sims.append(torch.nn.functional.cosine_similarity(e, target, dim=0).item())
        print(f"WER {wers[-1]:.2f} | similarity {sims[-1]:.2f} | {s!r} -> {hyp.strip()!r}")
    print(f"\nMean WER: {np.mean(wers):.3f} (lower is better)   Mean speaker similarity: {np.mean(sims):.3f}")
    return wers, sims
