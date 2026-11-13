"""Full pipeline: text -> phrases -> FastSpeech-style Transformer -> Diffusion (DDIM) -> GAN -> joined waveform.
(Falls back to the baseline fixed-length Transformer when no FastSpeech checkpoint exists.)
The AE cleans reference clips before the VAE turns them into a speaker embedding."""
import os
import re
import math
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
        words = [w[i:i + max_chars] for w in clean_text(phrase).split() for i in range(0, len(w), max_chars)]
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


MAX_FRAMES = 128        # the models were trained on 128-frame clips; past about 136 the decoder output decays into a hum
CONTEXT_WORDS = 1       # neighbouring words spoken with each part (and cut away) so the joins sound continuous
MAX_TEXT_CHARS = 240    # the text encoder's positional encoding covers 256 characters (window = chunk + context)
CONTEXT_RESERVE = 56    # frames kept free for that context when a chunk is split (a window's first word is
                        # spoken longer, because the model puts the leading silence on it)


def split_by_frames(chunk, durations, limit=MAX_FRAMES):
    """Split a chunk at word boundaries so that no part needs more than `limit` predicted frames.
    durations: predicted frames per character of `chunk`. Returns [chunk] when it already fits."""
    total = sum(durations)
    if total <= limit or " " not in chunk:
        return [chunk]
    target = total / math.ceil(total / limit)                     # even parts, each under the limit
    parts, cur, cur_frames, pos = [], [], 0, 0
    for word in chunk.split(" "):
        frames = sum(durations[pos:pos + len(word) + 1])          # the word plus the space after it
        pos += len(word) + 1
        if cur and cur_frames + frames / 2 > target:
            parts.append(" ".join(cur)); cur, cur_frames = [], 0
        cur.append(word); cur_frames += frames
    parts.append(" ".join(cur))
    return parts


def trim_silence(w, top_db=35, keep_ms=60, left=True, right=True):
    """Trim leading/trailing silence but keep a short margin, so word endings and soft
    final consonants (s, t, f) are not cut off. left/right: which ends to trim."""
    import librosa
    _, (a, b) = librosa.effects.trim(w, top_db=top_db)
    keep = int(SR * keep_ms / 1000)
    return w[max(0, a - keep // 2) if left else 0: min(len(w), b + keep) if right else len(w)]


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
    def _encode(self, text, speaker_emb, pace):
        """Encoder output (1, L, D) and predicted frames per character for `text`, at its own length (no
        padding, so the durations are exactly the ones the plan is built on)."""
        ids, mask = encode_text(text, self.meta["char2id"], max_len=len(text))
        ids, mask = torch.from_numpy(ids)[None].to(device), torch.from_numpy(mask)[None].to(device)
        h = self.tts.encode(ids, mask, speaker_emb[None])
        return h, self.tts.predict_durations(h, mask, pace)[0].tolist()

    @torch.no_grad()
    def _plan(self, chunks, speaker_emb, pace, limit=MAX_FRAMES):
        """Turn phrase chunks into segments the models can say in one go.
        A chunk that would run past `limit` frames is split at word boundaries into parts. Each part is then
        spoken together with up to CONTEXT_WORDS neighbouring words on each side, and only the part's own
        frames are kept (cut in the middle of the word gaps), so a join sounds like the middle of a sentence
        rather than the end of one and the start of another.
        Returns [(window_text, frame_start, frame_end, pause_after, (joined_left, joined_right), (h, durations))]."""
        segs = []
        for chunk, pause in chunks:
            h, dur = self._encode(chunk, speaker_emb, pace)
            reserve = min(int(CONTEXT_RESERVE * max(pace, 1.0)), limit // 2)   # slower speech: longer context words
            parts = split_by_frames(chunk, dur, limit - reserve) if sum(dur) > limit else [chunk]
            if len(parts) == 1:
                segs.append((chunk, 0, sum(dur), pause, (False, False), (h, dur)))
                continue
            words, a = chunk.split(" "), 0
            for i, part in enumerate(parts):
                n = part.count(" ") + 1                       # words in this part
                cl, cr = min(CONTEXT_WORDS, a), min(CONTEXT_WORDS, len(words) - a - n)
                while True:
                    ws = words[a - cl:a + n + cr]
                    window = " ".join(ws)
                    wh, wdur = self._encode(window, speaker_emb, pace)
                    if sum(wdur) <= limit or cl + cr == 0:
                        break
                    if cl >= cr: cl -= 1                      # too long: give up context, largest side first
                    else: cr -= 1
                if sum(wdur) > limit and " " in part:            # too long even alone: split this part again
                    segs += self._plan([(part, pause and i == len(parts) - 1)], speaker_emb, pace, limit)
                    a += n
                    continue
                cs = len(" ".join(ws[:cl])) + 1 if cl else 0       # first character of the part inside the window
                ce = cs + len(part)
                fs = sum(wdur[:cs - 1]) + wdur[cs - 1] // 2 if cl else 0           # middle of the gap before
                fe = sum(wdur[:ce]) + wdur[ce] // 2 if cr else sum(wdur)            # middle of the gap after
                # a side that lost its context is a plain cut: it gets trimmed like a chunk end
                segs.append((window, fs, fe, pause and i == len(parts) - 1, (cl > 0, cr > 0), (wh, wdur)))
                a += n
        return segs

    @torch.no_grad()
    def text_to_mel(self, text, speaker=None, speaker_emb=None, chunk_chars=None, pace=1.0, seed=0):
        """Returns [(mel (80, T) numpy, pause_after, (joined_left, joined_right))] per segment, before the
        vocoder. A joined side continues straight into its neighbour (no silence to trim there)."""
        if speaker_emb is None:
            speaker_emb = self.speaker_mean[speaker or self.meta["train_speakers"][0]]
        speaker_emb = speaker_emb.to(device)
        max_chars = chunk_chars or (200 if self.kind == "fastspeech" else self.meta["chunk_chars"])   # length is handled in _plan
        max_chars = min(max_chars, MAX_TEXT_CHARS)
        chunks = split_text(text, max_chars)
        if not chunks:
            return []
        if self.kind == "fastspeech":
            segs = self._plan(chunks, speaker_emb, pace)
            # decode every window with exactly the durations the plan was built on (no padding in the encoder)
            T = max(min(sum(d), 1024) for *_, (_, d) in segs)
            T += (-T) % 4                                      # the diffusion U-Net halves time twice
            coarse, lengths = [], []
            for *_, (h, d) in segs:
                d = torch.tensor([d], device=device, dtype=torch.long)
                m, fm = self.tts.decode(h, d, speaker_emb[None], max_len=T)
                coarse.append(m); lengths.append(int(fm.sum()))
            coarse = torch.cat(coarse).unsqueeze(1)            # (N, 1, 80, T), silence after each window
        else:
            segs = [(c, 0, None, pause, (False, False), None) for c, pause in chunks]
            enc = [encode_text(c, self.meta["char2id"]) for c, _ in chunks]
            ids = torch.from_numpy(np.stack([e[0] for e in enc])).to(device)
            mask = torch.from_numpy(np.stack([e[1] for e in enc])).to(device)
            coarse = self.tts(ids, mask, speaker_emb.unsqueeze(0).expand(len(chunks), -1).contiguous())
            lengths = [coarse.size(-1)] * len(chunks)
        spk = speaker_emb.unsqueeze(0).expand(len(segs), -1).contiguous()
        strength = self.strength if self.kind == "fastspeech" else 1.0
        mel = ddim_sample(self.diff, self.sched, coarse, spk, self.steps, self.cfg, seed=seed, strength=strength) \
            if self.use_diffusion else coarse
        out = []
        for j, (_, fs, fe, has_pause, joined, _) in enumerate(segs):
            n = lengths[j]
            fe = n if fe is None else min(fe, n)
            out.append((mel[j, 0, :, fs:fe].float().cpu().numpy(), has_pause, joined))
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
        for m, has_pause, (jl, jr) in mels:
            m = np.pad(m, ((0, 0), (2, tail)))
            x = torch.from_numpy(m)[None, None].to(device)
            w = self.G(x).squeeze().cpu().numpy()
            if jl:                                    # joined sides: cut exactly at the frame, keep no padding
                w = w[2 * HOP:]
            if jr:
                w = w[:len(w) - tail * HOP]
            t = trim_silence(w, trim_db, left=not jl, right=not jr)
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
