"""Export the five trained models to ONNX for the in-browser app (frontend/models/).

Run from the repository folder:  python tools/export_onnx.py
Writes five .onnx files plus frontend/models/pipeline.json (vocabulary, speaker voices, diffusion schedule,
mel filterbank) - everything the browser needs besides the models themselves.

FastSpeech is split in two so the browser can stretch each character by its predicted duration in between:
  tts_encoder   characters + voice -> character encodings + log durations
  tts_decoder   stretched encodings + voice -> mel spectrogram (0..1)
The diffusion refiner is one denoising step (the browser runs the few steps of shallow diffusion).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vc_lib.common import SR, N_FFT, HOP, N_MELS, SEG, paths, load_meta, tload  # noqa: E402
from vc_lib.train_ae import load_ae  # noqa: E402
from vc_lib.train_vae import load_vae  # noqa: E402
from vc_lib.train_fastspeech import load_fastspeech  # noqa: E402
from vc_lib.train_diffusion import load_diffusion, Schedule  # noqa: E402
from vc_lib.train_gan import load_generator  # noqa: E402

OUT = ROOT / "frontend" / "models"


class TTSEncoder(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.fs = fs

    def forward(self, ids, spk):                                   # ids (1, L) int64, spk (1, 64)
        mask = torch.ones_like(ids, dtype=torch.float32)
        h = self.fs.encode(ids, mask, spk)
        return h, self.fs.duration(h, mask)                        # (1, L, 256), (1, L) log(1 + frames)


class TTSDecoder(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.fs = fs

    def forward(self, x, spk):                                     # x (1, T, 256) already stretched
        x = self.fs.dec_pos(x) + self.fs.dec_speaker(spk).unsqueeze(1)
        y = self.fs.mel_decoder(x)
        return torch.sigmoid(self.fs.output_proj(y)).transpose(1, 2)   # (1, 80, T)


class VoiceEncoder(nn.Module):
    def __init__(self, ae, vae):
        super().__init__()
        self.ae, self.vae = ae, vae

    def forward(self, windows):                                    # (N, 1, 80, 128) mel windows in 0..1
        return self.vae.encode(self.ae(windows))[0]                # (N, 64); the browser averages them


def export(model, args, name, inputs, outputs, shapes):
    """torch.export-based exporter (the older tracer fixes attention lengths), then one self-contained file."""
    import onnx
    path = OUT / f"{name}.onnx"
    torch.onnx.export(model, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_shapes=shapes, dynamo=True, opset_version=18)
    m = onnx.load(str(path))                                       # pulls in any separate weights file
    data = Path(str(path) + ".data")
    onnx.save(m, str(path), save_as_external_data=False)
    data.unlink(missing_ok=True)
    print(f"{name}.onnx  {path.stat().st_size / 1e6:.1f} MB")


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)                 # the fused attention kernel has no ONNX form
    root, p = str(ROOT), paths(str(ROOT))
    meta = load_meta(p)
    fs, diff, gen = load_fastspeech(root).cpu().eval(), load_diffusion(root).cpu().eval(), load_generator(root).cpu().eval()
    ae, vae = load_ae(root).cpu().eval(), load_vae(root).cpu().eval()
    spk = torch.randn(1, 64)

    Dim = torch.export.Dim
    L, T, K, N = Dim("L", min=2, max=255), Dim("T", min=2, max=4095), Dim("K", min=1, max=1024), Dim("N", min=1, max=256)
    export(TTSEncoder(fs), (torch.randint(2, 30, (1, 20)), spk), "tts_encoder", ["ids", "spk"], ["h", "logd"], ({1: L}, None))
    export(TTSDecoder(fs), (torch.randn(1, 64, 256), spk), "tts_decoder", ["x", "spk"], ["mel"], ({1: T}, None))
    export(diff, (torch.randn(1, 1, 80, 64), torch.tensor([20]), torch.rand(1, 1, 80, 64), spk), "diffusion",
           ["x", "t", "cond", "spk"], ["eps"], ({3: 4 * K}, None, {3: 4 * K}, None))   # length: multiple of 4
    export(gen, (torch.rand(1, 1, 80, 64),), "vocoder", ["mel"], ["wav"], ({3: T},))
    export(VoiceEncoder(ae, vae), (torch.rand(2, 1, 80, SEG),), "voice_encoder", ["windows"], ["mu"], ({0: N},))

    import librosa
    sched = Schedule(200)
    speaker_mean = tload(p["speaker_mean"])
    info_file = ROOT / "data" / "speaker_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    voices = [{"id": s, "seen_in_training": s in meta["train_speakers"], **info.get(s, {}),
               "embedding": [round(float(v), 5) for v in speaker_mean[s]]}
              for s in meta["train_speakers"] + meta["holdout"] if s in speaker_mean]
    cfg = {
        "sample_rate": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "seg": SEG,
        "db_range": meta["db_range"], "char2id": meta["char2id"], "unk_id": 1, "max_chars": 50,
        "diffusion": {"T": sched.T, "alpha_bar": [float(a) for a in sched.alpha_bar.cpu()], "strength": 0.1, "steps": 50},
        "mel_basis": np.round(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS), 7).tolist(),
        "voices": voices,
    }
    (OUT / "pipeline.json").write_text(json.dumps(cfg, separators=(",", ":")))
    print(f"pipeline.json  {(OUT / 'pipeline.json').stat().st_size / 1e6:.1f} MB, {len(voices)} voices")


if __name__ == "__main__":
    main()
