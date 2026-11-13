# Synthetic Voice Cloning TTS

Text-to-speech with voice cloning, built from five models trained from scratch on 80 speakers of the
VCTK corpus. Type text, choose a voice, and get speech; a short recording of a new voice can be used too.

| Result (7 test sentences, Whisper `small`) | Word error rate | Sentences perfect | Voice similarity |
| --- | --- | --- | --- |
| Original design (fixed-slot Transformer) | 0.960 | 0% | 0.629 |
| **Final pipeline** | **0.083** | **71%** | **0.744** |

On 30 validation phrases the generated speech scores WER 0.119, against 0.172 for the real recordings
of the same phrases. On 2 CPU cores it speaks 9 to 60 times faster than real time.

## How it works

```
text ──► 3 FastSpeech-style Transformer ──► 4 Diffusion refiner ──► 5 GAN vocoder ──► audio
              (timing + spectrogram)          (shallow, strength 0.1)   (spectrogram to audio)
                     ▲
voice clip ──► 1 Denoising autoencoder ──► 2 Speaker VAE ──► speaker embedding (64 numbers)
```

| Model | Role | Checkpoint |
| --- | --- | --- |
| 1. Denoising autoencoder | Cleans the voice clip | `checkpoints/ae_best.pt` |
| 2. Speaker VAE | Turns a voice into 64 numbers (98.6% speaker accuracy) | `checkpoints/vae_best.pt` |
| 3. FastSpeech-style Transformer | Predicts how long each character lasts and builds the spectrogram | `checkpoints/fastspeech_best.pt` |
| 4. Diffusion refiner | Lightly sharpens the spectrogram | `checkpoints/diffusion_best.pt` |
| 5. GAN vocoder | Turns the spectrogram into a waveform | `checkpoints/gan_best.pt` |

`checkpoints/speaker_mean.pt` holds one embedding per built-in speaker. `transformer_best.pt` and
`baseline_diffusion_best.pt` are the original design, kept for comparison.

## Use it

Requires Python 3.11 and PyTorch (a GPU is optional).

```bash
pip install torch torchaudio librosa soundfile num2words
```

```python
import sys; sys.path.insert(0, ".")          # run from the repository folder
from vc_lib.infer import VoicePipeline
import soundfile as sf

pipe = VoicePipeline(".")                     # loads the five models
speaker = pipe.meta["train_speakers"][0]      # a built-in voice, e.g. "p225"
audio = pipe.synthesize("The meeting has been moved to Thursday afternoon.", speaker=speaker)
sf.write("out.wav", audio, 22050)

# clone a voice from a 5 to 20 second recording
emb = pipe.embed_reference(["my_voice.wav"])
sf.write("mine.wav", pipe.synthesize("Hello, this is my voice.", speaker_emb=emb), 22050)
```

`synthesize` also takes `pace` (below 1 speaks faster, above 1 slower).

## Repository layout

| Path | Contents |
| --- | --- |
| `vc_lib/` | Models, data preparation, training and inference code |
| `notebooks/` | Training and evaluation notebooks, run in order (see below) |
| `tests/smoke_test.py` | End-to-end test on synthetic data (`python tests/smoke_test.py`) |
| `data/meta.json`, `data/speakers.json` | Settings and speaker list the models need |
| `checkpoints/` | Trained models |
| `backend/` | Voice Cloner web app: API for the models (Hugging Face Space) |
| `frontend/` | Voice Cloner web app: the website (Netlify) |

## Training from scratch

The training data is not in this repository (about 30 GB). Run the notebooks in order; each one
starts with a cell that sets `ROOT` to the project folder.

| Notebook | Step |
| --- | --- |
| `00_data_prep` | Download VCTK, cut phrases with Whisper word timings, build the cache |
| `01_autoencoder` | Model 1 |
| `02_speaker_vae` | Model 2 |
| `03b_fastspeech` | Model 3 (final) — `03_transformer` is the original design |
| `04b_diffusion` | Model 4 on the FastSpeech output — `04_diffusion` is the original |
| `05_gan_vocoder` | Model 5 |
| `06b_demo_pipeline` | Listen and evaluate the full pipeline |
| `07_shallow_diffusion` | Diffusion strength study |
| `08_diffusion_predicted` | Diffusion retrained on predicted durations (did not help) |

## Limitations

- Audio can sound slightly metallic (small vocoder).
- New voices are approximate: similarity 0.58 for unseen speakers against 0.74 to 0.78 for training voices.
- The model reads letters, not phonemes, so some names are mispronounced.

## References

FastSpeech (Ren et al., 2019) · Glow-TTS monotonic alignment search (Kim et al., 2020) ·
RAD-TTS alignment (Shih et al., 2021) · DiffSinger shallow diffusion (Liu et al., 2022) ·
VCTK corpus (Yamagishi et al.) · Whisper (Radford et al., 2022).

Clone only voices you own or have permission to use.
