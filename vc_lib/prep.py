"""Step 0: build the dataset.
prepare_shards(): VCTK -> trimmed, peak-normalised utterances -> Whisper word timestamps -> short
                  segments (<= ~1.44 s, <= 60 chars) whose text matches their audio. One shard per speaker
                  (resumable: finished speakers are skipped after a Colab disconnect).
build_cache():    shards -> measured dB range, mels, text ids, train/val/holdout split, vocoder crops.
"""
import os, json, difflib, random
import numpy as np
import torch
from tqdm.auto import tqdm
from .common import (SR, HOP, SEG, WAV_LEN, N_MELS, MAX_TEXT_LEN, paths, tload, safe_save, norm_wav, raw_db,
                     wav_to_mel, fit_frames, fit_wav, to_int16, from_int16, clean_text, build_vocab,
                     encode_text)

MAX_SEG_SEC = WAV_LEN / SR - 0.05     # leave room for edge padding


# ---------------- pure helpers (unit-tested) ----------------
def group_words(words, max_sec=MAX_SEG_SEC, max_chars=MAX_TEXT_LEN):
    """words: list of (start, end, text). Greedy grouping into segments by duration and length."""
    groups, cur = [], []
    for k, (s, e, _) in enumerate(words):
        if cur:
            s0 = words[cur[0]][0]
            txt = " ".join(words[j][2] for j in cur + [k])
            if e - s0 > max_sec or len(txt) > max_chars:
                groups.append(cur)
                cur = []
        cur.append(k)
    if cur:
        groups.append(cur)
    return groups


def segment_bounds(words, group, total_sec, edge_pad=0.03):
    """Pad each side by edge_pad, but never past halfway to the neighbouring word (no bleed)."""
    s0, e0 = words[group[0]][0], words[group[-1]][1]
    a = s0 - edge_pad
    if group[0] > 0:
        a = max(a, (words[group[0] - 1][1] + s0) / 2)
    b = e0 + edge_pad
    if group[-1] + 1 < len(words):
        b = min(b, (e0 + words[group[-1] + 1][0]) / 2)
    return max(a, 0.0), min(b, total_sec)


def words_from_whisper(segments):
    out = []
    for seg in segments:
        for w in (seg.words or []):
            t = clean_text(w.word)
            if t and w.end > w.start:
                out.append((float(w.start), float(w.end), t))
    return out


def cut_utterance(y, words, edge_pad=0.03, min_sec=0.2):
    """y: normalised float wav at SR. Returns list of (segment_wav, text)."""
    total = len(y) / SR
    out = []
    for g in group_words(words):
        a, b = segment_bounds(words, g, total, edge_pad)
        if b - a < min_sec or b - a > WAV_LEN / SR:
            continue                      # too short to be useful / single word too long for the window
        text = " ".join(words[j][2] for j in g)
        out.append((y[int(a * SR):int(b * SR)], text))
    return out


HOLDOUT_SPEAKERS = ["p263", "p264", "p265", "p266"]   # unseen voices, identical in every run


def choose_speakers(available, num_speakers=40, num_holdout=4):
    """Training voices = first (num_speakers - num_holdout) non-holdout speakers in sorted order.
    Holdout is ALWAYS p263-p266, so results from different runs stay comparable."""
    holdout = HOLDOUT_SPEAKERS[:num_holdout]
    missing = [h for h in holdout if h not in available]
    assert not missing, f"Holdout speakers not in dataset: {missing}"
    train = [x for x in sorted(available) if x not in holdout][:num_speakers - num_holdout]
    return {"selected": train + holdout, "holdout": holdout}


# ---------------- step A: shards ----------------
def _find_text_key(ds):
    for key in ["text", "transcription", "sentence", "normalized_text"]:
        if key in ds.column_names:
            return key
    raise KeyError(f"No text column found. Columns: {ds.column_names}")


def prepare_shards(root, num_speakers=40, num_holdout=4, whisper_size="small",
                   whisper_device="cuda", whisper_compute="float16", min_match=0.9, max_utts_per_speaker=None):
    import librosa
    from datasets import load_dataset
    from faster_whisper import WhisperModel

    p = paths(root)
    ds = load_dataset("CSTR-Edinburgh/vctk", split="train", revision="refs/convert/parquet")
    text_key = _find_text_key(ds)
    spk_col = list(ds["speaker_id"])
    txt_col = list(ds[text_key])

    if os.path.exists(p["speakers"]):
        info = json.load(open(p["speakers"]))
        print("Using existing speaker selection:", info)
    else:
        with_text = sorted({s for s, t in zip(spk_col, txt_col) if t and str(t).strip()})
        info = choose_speakers(with_text, num_speakers, num_holdout)
        json.dump(info, open(p["speakers"], "w"), indent=1)
    print(f"{len(info['selected'])} speakers, holdout (unseen) = {info['holdout']}")

    whisper = WhisperModel(whisper_size, device=whisper_device, compute_type=whisper_compute)

    for spk in info["selected"]:
        shard_path = os.path.join(p["shards"], f"{spk}.pt")
        if os.path.exists(shard_path):
            continue
        idxs = [i for i, (s, t) in enumerate(zip(spk_col, txt_col)) if s == spk and t and str(t).strip()]
        if max_utts_per_speaker:
            idxs = idxs[:max_utts_per_speaker]
        seg_wavs, seg_texts, seg_utt, utt_wavs, kept, skipped = [], [], [], [], 0, 0
        for i in tqdm(idxs, desc=spk):
            ex = ds[i]
            y = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            y, _ = librosa.effects.trim(y, top_db=30)
            if len(y) < SR // 4:
                skipped += 1; continue
            y = norm_wav(y)                                    # the ONLY normalisation
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
            segments, _ = whisper.transcribe(y16, language="en", word_timestamps=True, beam_size=2)
            words = words_from_whisper(segments)
            heard = " ".join(w[2] for w in words)
            truth = clean_text(ex[text_key])
            if not words or difflib.SequenceMatcher(None, heard, truth).ratio() < min_match:
                skipped += 1; continue                         # audio and transcript disagree
            pieces = cut_utterance(y, words)
            if not pieces:
                skipped += 1; continue
            u = len(utt_wavs)
            utt_wavs.append(to_int16(y))
            for w, t in pieces:
                seg_wavs.append(to_int16(fit_wav(w))); seg_texts.append(t); seg_utt.append(u)
            kept += 1
        safe_save({"speaker": spk, "seg_wavs": torch.stack(seg_wavs) if seg_wavs else torch.zeros(0, WAV_LEN, dtype=torch.int16),
                   "seg_texts": seg_texts, "seg_utt": seg_utt, "utt_wavs": utt_wavs}, shard_path)
        print(f"{spk}: kept {kept} utterances -> {len(seg_texts)} segments (skipped {skipped})")
    print("All shards ready.")


# ---------------- step B: cache ----------------
def build_cache(root, val_frac=0.03, crops_per_utt=1, db_sample=600, seed=0):
    """Low-RAM build. Keeps only what training needs:
      cache.pt         segment mels (all speakers) + text ids + split labels + shard index
      gan_crops.pt     vocoder crops from TRAINING utterances (mels + int16 wavs)
      gan_holdout.pt   vocoder crops from the UNSEEN speakers (validation)
    Segment audio is not stored; it is read from the shards when needed (sanity checks).
    Split is by SENTENCE: both microphone copies of a sentence go to the same split."""
    p = paths(root)
    rng = random.Random(seed)
    info = json.load(open(p["speakers"]))
    holdout = set(info["holdout"])
    train_speakers = [s for s in info["selected"] if s not in holdout]
    shard_files = [os.path.join(p["shards"], f"{s}.pt") for s in info["selected"]]
    missing = [f for f in shard_files if not os.path.exists(f)]
    assert not missing, f"Missing shards (run prepare_shards first): {missing[:3]}..."

    # 1) sentence-level split and exact sizes
    shards, n_seg, n_tr, n_ho = {}, 0, 0, 0
    for s, f in tqdm(list(zip(info["selected"], shard_files)), desc="planning"):
        sh = tload(f)
        n_utt = len(sh["utt_wavs"])
        keys = [[] for _ in range(n_utt)]
        for t, u in zip(sh["seg_texts"], sh["seg_utt"]):
            keys[u].append(t)
        keys = [" ".join(k) for k in keys]
        if s in holdout:
            utt_split = ["holdout"] * n_utt
        else:
            key_split = {}
            for k in sorted(set(keys)):
                key_split[k] = "val" if rng.random() < val_frac else "train"
            utt_split = [key_split[k] for k in keys]
        shards[s] = (f, utt_split)
        n_seg += len(sh["seg_texts"])
        n_tr += crops_per_utt * sum(1 for x in utt_split if x == "train")
        n_ho += crops_per_utt * sum(1 for x in utt_split if x == "holdout")
        del sh

    # 2) dB range from real (non-padding) frames of training segments
    vals = []
    sample_files = [shards[s][0] for s in train_speakers]
    per_file = max(1, db_sample // max(1, len(sample_files)))
    for f in sample_files:
        sh = tload(f)
        n = len(sh["seg_texts"])
        for j in rng.sample(range(n), min(per_file, n)):
            d = raw_db(from_int16(sh["seg_wavs"][j]).numpy())
            vals.append(d[d > -99.0].ravel())
        del sh
    vals = np.concatenate(vals)
    db_range = [float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.9))]
    print(f"Measured dB range: {db_range[0]:.1f} .. {db_range[1]:.1f} dB")
    gb = (n_seg * N_MELS * SEG * 2 + (n_tr + n_ho) * (N_MELS * SEG * 2 + WAV_LEN * 2)) / 1e9
    print(f"Allocating {n_seg} segment mels + {n_tr} training crops + {n_ho} unseen-speaker crops (~{gb:.1f} GB RAM)")

    # 3) fill preallocated tensors
    mels = torch.empty(n_seg, N_MELS, SEG, dtype=torch.float16)
    tr_mels = torch.empty(n_tr, N_MELS, SEG, dtype=torch.float16)
    tr_wavs = torch.empty(n_tr, WAV_LEN, dtype=torch.int16)
    ho_mels = torch.empty(n_ho, N_MELS, SEG, dtype=torch.float16)
    ho_wavs = torch.empty(n_ho, WAV_LEN, dtype=torch.int16)
    texts, spks, split, seg_local = [], [], [], []
    i = c_tr = c_ho = 0
    for s in tqdm(info["selected"], desc="caching"):
        f, utt_split = shards[s]
        sh = tload(f)
        for j, t in enumerate(sh["seg_texts"]):
            w = from_int16(sh["seg_wavs"][j]).numpy()
            mels[i] = torch.from_numpy(fit_frames(wav_to_mel(w, db_range)))
            texts.append(t); spks.append(s); seg_local.append(j)
            split.append(utt_split[sh["seg_utt"][j]])
            i += 1
        for u, uw in enumerate(sh["utt_wavs"]):
            sp = utt_split[u]
            if sp not in ("train", "holdout"):
                continue
            y = from_int16(uw).numpy()
            if len(y) < WAV_LEN + HOP:
                y = fit_wav(y, WAV_LEN + HOP)
            m = wav_to_mel(y, db_range)
            max_start = m.shape[1] - SEG
            for _ in range(crops_per_utt):
                st = rng.randint(0, max(0, max_start))
                cm = torch.from_numpy(fit_frames(m[:, st:st + SEG]))
                cw = to_int16(fit_wav(y[st * HOP: st * HOP + WAV_LEN]))
                if sp == "train":
                    tr_mels[c_tr] = cm; tr_wavs[c_tr] = cw; c_tr += 1
                else:
                    ho_mels[c_ho] = cm; ho_wavs[c_ho] = cw; c_ho += 1
        del sh
    assert i == n_seg and c_tr == n_tr and c_ho == n_ho, (i, n_seg, c_tr, n_tr, c_ho, n_ho)

    train_texts = [t for t, sp in zip(texts, split) if sp == "train"]
    char2id = build_vocab(train_texts)
    enc = [encode_text(t, char2id) for t in texts]
    text_ids = torch.from_numpy(np.stack([e[0] for e in enc]))
    text_mask = torch.from_numpy(np.stack([e[1] for e in enc]))
    del enc

    assert safe_save({"mels": mels, "texts": texts, "spks": spks, "split": split, "seg_local": seg_local,
                      "text_ids": text_ids, "text_mask": text_mask}, p["cache"]), "cache save failed"
    del mels
    assert safe_save({"mels": tr_mels, "wavs": tr_wavs}, p["gan_crops"]), "crop save failed"
    del tr_mels, tr_wavs
    assert safe_save({"mels": ho_mels, "wavs": ho_wavs}, p["gan_holdout"]), "holdout save failed"
    del ho_mels, ho_wavs

    lens = [len(t) for t in train_texts]
    meta = {"db_range": db_range, "char2id": char2id, "vocab_size": len(char2id) + 2,
            "selected": info["selected"], "holdout": info["holdout"], "train_speakers": train_speakers,
            "chunk_chars": int(np.median(lens)),
            "counts": {k: split.count(k) for k in ("train", "val", "holdout")},
            "n_gan_crops": n_tr, "n_gan_holdout": n_ho}
    json.dump(meta, open(p["meta"], "w"), indent=1)
    print("Cache saved:", meta["counts"], "| vocoder crops:", n_tr, "| unseen crops:", n_ho,
          "| vocab:", meta["vocab_size"], "| median chars/segment:", meta["chunk_chars"])
    return meta
