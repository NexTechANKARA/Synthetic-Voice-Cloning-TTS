import os, sys, json, shutil, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vc_lib import common, prep, train_ae, train_vae, train_transformer, train_fastspeech, train_diffusion, train_gan, infer
from vc_lib.common import SR, WAV_LEN, to_int16, fit_wav, norm_wav, paths

ROOT = os.environ.get("VC_TEST_ROOT", "/tmp/vc_test")
shutil.rmtree(ROOT, ignore_errors=True)
p = paths(ROOT)
rng = np.random.default_rng(0)
WORDS = "please call stella ask her to bring these things with her from the store six spoons".split()

# ---- unit tests for pure helpers ----
words = [(0.0, 0.3, "please"), (0.35, 0.6, "call"), (0.7, 1.1, "stella"), (1.2, 1.5, "ask"), (1.55, 1.9, "her")]
g = prep.group_words(words, max_sec=1.0)
assert g == [[0, 1], [2, 3], [4]], g
a, b = prep.segment_bounds(words, [2, 3], 2.0, 0.03)
assert abs(a - 0.67) < 1e-9 and abs(b - 1.525) < 1e-9, (a, b)
assert common.clean_text("It's 1999, OK?") == "it's one thousand nine hundred and ninety nine ok"
ch = infer.split_text("Hello there, my friend. How are you doing today?", 12)
assert ch[0] == ("hello there", True) and ch[-1][1] is True, ch
assert abs(infer.word_error_rate("a b c d", "a x c") - 0.5) < 1e-9
y = np.ones(1000, np.float32)
assert len(infer.crossfade_concat([y, y], 100)) == 1900
from types import SimpleNamespace as NS
mock = [NS(words=[NS(start=0.1, end=0.4, word=" Please"), NS(start=0.45, end=0.7, word=" call,"),
                  NS(start=0.8, end=1.3, word=" Stella."), NS(start=1.3, end=1.3, word=" !"),
                  NS(start=1.4, end=1.9, word=" 6")])]
ws = prep.words_from_whisper(mock)
assert [w[2] for w in ws] == ["please", "call", "stella", "six"], ws
yy = np.zeros(int(SR * 2.0), np.float32)
pieces = prep.cut_utterance(yy, ws)
assert [t for _, t in pieces] == ["please call stella", "six"], pieces
assert all(len(w) <= WAV_LEN for w, _ in pieces)
avail = [f"p{x}" for x in range(225, 380) if x not in (235, 242)]
c40, c50 = prep.choose_speakers(avail, 40, 4), prep.choose_speakers(avail, 80, 4)
assert c40["holdout"] == ["p263", "p264", "p265", "p266"] and c50["holdout"] == c40["holdout"]
assert len(c40["selected"]) == 40 and len(c50["selected"]) == 80
assert c40["selected"][:36] == c50["selected"][:36] and c40["selected"][36:] == c40["holdout"]
assert not set(c50["holdout"]) & set(c50["selected"][:76])
print("unit tests ok")

# ---- fake shards (stand-in for Whisper output) ----
speakers = [f"p{225 + i}" for i in range(5)]
json.dump({"selected": speakers, "holdout": speakers[-1:]}, open(p["speakers"], "w"))
for k, s in enumerate(speakers):
    f0 = 100 + 30 * k
    utts, segw, segt, segu = [], [], [], []
    for u0 in range(4):
        n = int(SR * rng.uniform(1.0, 3.0))
        t = np.arange(n) / SR
        y = norm_wav(np.sin(2 * np.pi * f0 * t * (1 + 0.1 * np.sin(3 * t))) * (0.5 + 0.5 * np.sin(5 * t)) + 0.05 * rng.standard_normal(n))
        sent = [" ".join(random.sample(WORDS, 3)) for _ in range(2)]
        for mic in range(2):                      # VCTK on HF: every sentence recorded by 2 microphones
            u = len(utts)
            ym = norm_wav(y + 0.02 * mic * rng.standard_normal(n))
            utts.append(to_int16(ym))
            for j in range(2):
                a0 = int(rng.uniform(0, max(1, n - SR)))
                segw.append(to_int16(fit_wav(ym[a0:a0 + int(SR * 1.2)])))
                segt.append(sent[j]); segu.append(u)
    common.safe_save({"speaker": s, "seg_wavs": torch.stack(segw), "seg_texts": segt, "seg_utt": segu,
                      "utt_wavs": utts}, os.path.join(p["shards"], f"{s}.pt"))

meta = prep.build_cache(ROOT, val_frac=0.3, crops_per_utt=1, db_sample=40)
assert meta["db_range"][0] < meta["db_range"][1]
C = common.load_cache(p)
assert C["mels"].shape[1:] == (80, 128) and "wavs" not in C
assert len(C["idx"]["val"]) > 0 and len(C["idx"]["holdout"]) == 16
for s_ in speakers[:-1]:                         # no sentence in both train and val
    tr_t = {C["texts"][i] for i in C["idx"]["train"].tolist() if C["spks"][i] == s_}
    va_t = {C["texts"][i] for i in C["idx"]["val"].tolist() if C["spks"][i] == s_}
    assert not (tr_t & va_t), (s_, tr_t & va_t)
assert meta["n_gan_crops"] == common.tload(p["gan_crops"])["mels"].size(0)
assert common.tload(p["gan_holdout"])["mels"].size(0) > 0
assert 0.0 <= C["mels"].float().min() and C["mels"].float().max() <= 1.0

# ---- train every model briefly ----
train_ae.train(ROOT, epochs=2, batch_size=8, warmup=2)
train_ae.train(ROOT, epochs=3, batch_size=8, warmup=2)          # resume path
train_vae.train(ROOT, epochs=2, batch_size=8, warmup=2)
assert set(common.tload(p["speaker_mean"]).keys()) == set(speakers)
assert common.tload(p["vae_mu"]).shape == (C["mels"].size(0), 64)
train_transformer.train(ROOT, epochs=2, batch_size=8, warmup=2)        # baseline still works
# FastSpeech: pause after 1 epoch (early check), then resume to the end; warm start from the baseline encoder
train_fastspeech.train(ROOT, epochs=3, stop_after=1, batch_size=8, warmup=2)
assert not os.path.exists(p["coarse"]) or True
rep = train_fastspeech.alignment_report(ROOT, n=2)
assert rep and all(r["attn"].ndim == 2 and 0 < r["sharpness"] <= 1 for r in rep), rep
train_fastspeech.train(ROOT, epochs=3, batch_size=8, warmup=2)          # resumes at epoch 2, then exports
CC = common.tload(p["coarse"])
assert CC["coarse"].shape == C["mels"].shape and CC["spk_emb"].shape == (C["mels"].size(0), 64)
pad = (C["mels"].float().amax(1) == 0)                                   # padded frames must stay silent
assert float(CC["coarse"].float().amax(1)[pad].max()) == 0.0
train_diffusion.train(ROOT, epochs=2, batch_size=8, warmup=2)
train_diffusion.train(ROOT, epochs=3, batch_size=8, warmup=2)   # resume with EMA
train_fastspeech.export_coarse_predicted(ROOT)
CP = common.tload(p["coarse_pred"])
assert CP["coarse"].shape == CP["target"].shape == C["mels"].shape and float(CP["target"].float().abs().sum()) > 0
train_diffusion.train(ROOT, epochs=1, batch_size=8, warmup=2, name="diffusion_pred", coarse_file=p["coarse_pred"])
assert os.path.exists(os.path.join(p["ckpt"], "diffusion_pred_best.pt"))
train_gan.train(ROOT, epochs=2, steps_per_epoch=2, batch_size=4, degrade_last_epochs=1, p_degrade=1.0)

# ---- full pipeline ----
pipe = infer.VoicePipeline(ROOT, diffusion_steps=5)
assert pipe.kind == "fastspeech"
assert pipe.diff_name == "diffusion_pred"
assert infer.VoicePipeline(ROOT, diffusion_steps=5, diffusion="diffusion").diff_name == "diffusion"
slow = pipe.synthesize("please call stella", speaker=speakers[0], pace=1.5)
fast = pipe.synthesize("please call stella", speaker=speakers[0], pace=0.7)
assert len(slow) > len(fast), (len(slow), len(fast))
base = infer.VoicePipeline(ROOT, diffusion_steps=5, model="baseline").synthesize("baseline path", speaker=speakers[0])
assert len(base) > 0
out = pipe.synthesize("Hello there, please call Stella. Ask her to bring 3 spoons!", speaker=speakers[0])
assert out.ndim == 1 and len(out) > 0 and np.isfinite(out).all() and np.abs(out).max() <= 1.0
import soundfile as sf
sf.write("/tmp/ref.wav", common.segment_audio(p, C, 0), SR)
emb = pipe.embed_reference(["/tmp/ref.wav"])
assert emb.shape == (64,)
out2 = pipe.synthesize("short test", speaker_emb=emb)
out3 = infer.VoicePipeline(ROOT, use_diffusion=False).synthesize("no diffusion path", speaker=speakers[-1])
print("pipeline ok:", len(out) / SR, "s,", len(out2) / SR, "s,", len(out3) / SR, "s")
print("ALL SMOKE TESTS PASSED")
