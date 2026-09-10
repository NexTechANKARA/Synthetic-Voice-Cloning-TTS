import os
import io
import math
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
SEGMENT_FRAMES = 128
LATENT_DIM = 64
NUM_SPEAKERS = 10

SELECTED_SPEAKERS = ["p225", "p226", "p227", "p228", "p229",
                     "p230", "p231", "p232", "p233", "p234"]

VOCAB_CHARS = [' ', '!', "'", ',', '-', '.', '?'] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
PAD_ID = 0
UNK_ID = 1
char2id = {c: i + 2 for i, c in enumerate(VOCAB_CHARS)}
id2char = {i + 2: c for i, c in enumerate(VOCAB_CHARS)}
id2char[PAD_ID] = '<PAD>'
id2char[UNK_ID] = '<UNK>'
VOCAB_SIZE = len(char2id) + 2  # 35
MAX_TEXT_LEN = 60

# --- Audio utilities ---
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

inv_mel_scale = T.InverseMelScale(
    n_stft=N_FFT // 2 + 1,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE
)

griffin_lim = T.GriffinLim(
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_iter=32
)

import soundfile as sf

def audio_bytes_to_mel(audio_bytes: bytes, target_frames: int = SEGMENT_FRAMES):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    waveform = torch.from_numpy(data).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[1] < waveform.shape[0]:
        waveform = waveform.t()
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Compute mel spectrogram
    mel = mel_transform(waveform).squeeze(0) # (80, T)
    mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = torch.clamp(mel_norm, 0.0, 1.0)

    # Pad or crop to target_frames
    T_curr = mel_norm.shape[1]
    if T_curr >= target_frames:
        mel_cropped = mel_norm[:, :target_frames]
    else:
        pad_len = target_frames - T_curr
        mel_cropped = F.pad(mel_norm, (0, pad_len), mode="constant", value=0.0)

    return mel_cropped.unsqueeze(0).unsqueeze(0)  # (1, 1, 80, 128)

def mel_to_wav_bytes(mel_tensor: torch.Tensor, watermark: bool = True, speaker_id: str = "synthetic") -> bytes:
    """Convert mel tensor (80, 128) in [0, 1] range back to WAV audio bytes using Griffin-Lim with optional ethical watermarking."""
    with torch.no_grad():
        if mel_tensor.dim() == 4:
            mel_tensor = mel_tensor.squeeze(0).squeeze(0)
        elif mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.squeeze(0)

        mel_norm = torch.clamp(mel_tensor, 0.0, 1.0)
        mel_db = mel_norm * 80.0 - 80.0
        mel_power = torch.pow(10.0, mel_db / 10.0)

        lin_spec = inv_mel_scale(mel_power)
        wav = griffin_lim(lin_spec).cpu().numpy()

        # Normalize waveform
        max_val = np.max(np.abs(wav))
        if max_val > 1e-6:
            wav = wav / max_val * 0.95
        wav_int16 = (wav * 32767).astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, SAMPLE_RATE, wav_int16)
        raw_bytes = buf.getvalue()

        if watermark:
            try:
                from ethical_safeguards import watermarker
                return watermarker.embed_watermark(raw_bytes, speaker_id=speaker_id)
            except Exception:
                pass
        return raw_bytes

def encode_text(text: str, max_len: int = MAX_TEXT_LEN):
    tokens = [char2id.get(c, UNK_ID) for c in text.lower()][:max_len]
    mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    tokens = tokens + [PAD_ID] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


# ==================== Model Architectures ====================

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SpeakerVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_speakers=NUM_SPEAKERS):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.flatten_dim = 64 * 10 * 16
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1), nn.Sigmoid(),
        )
        self.speaker_classifier = nn.Linear(latent_dim, num_speakers)

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std * temperature

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 10, 16)
        return self.dec_conv(h)

    def forward(self, x, temperature=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, temperature)
        return self.decode(z), mu, logvar, self.speaker_classifier(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTTS(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=4,
                 enc_layers=3, dec_layers=3, n_mels=N_MELS, seg_frames=SEGMENT_FRAMES,
                 speaker_dim=LATENT_DIM):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        self.mel_queries = nn.Parameter(torch.randn(seg_frames, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.mel_decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        self.output_proj = nn.Linear(d_model, n_mels)

    def forward(self, text_ids, text_mask, speaker_emb):
        B = text_ids.size(0)
        text_emb = self.pos_enc(self.text_embed(text_ids))
        key_padding_mask = (text_mask == 0)
        enc_out = self.text_encoder(text_emb, src_key_padding_mask=key_padding_mask)

        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        enc_out = enc_out + spk

        queries = self.mel_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.mel_decoder(tgt=queries, memory=enc_out, memory_key_padding_mask=key_padding_mask)

        mel_out = self.output_proj(dec_out)
        mel_out = torch.sigmoid(mel_out).transpose(1, 2).unsqueeze(1)
        return mel_out


# --- Model Registry & Loader ---
class ModelManager:
    def __init__(self, models_dir: str = "Trained/Trained_models"):
        self.models_dir = models_dir
        self.device = torch.device("cpu")
        self.autoencoder = None
        self.speaker_vae = None
        self.transformer_tts = None
        self.load_models()

    def load_models(self):
        # 1. Denoising Autoencoder
        ae_path = os.path.join(self.models_dir, "autoencoder_full.pt")
        self.autoencoder = DenoisingAutoencoder().to(self.device)
        if os.path.exists(ae_path):
            ckpt = torch.load(ae_path, map_location=self.device, weights_only=False)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            self.autoencoder.load_state_dict(state)
            self.autoencoder.eval()
            print(f"Loaded Autoencoder from {ae_path}")

        # 2. Speaker VAE
        vae_path = os.path.join(self.models_dir, "speaker_vae_full.pt")
        self.speaker_vae = SpeakerVAE().to(self.device)
        if os.path.exists(vae_path):
            ckpt = torch.load(vae_path, map_location=self.device, weights_only=False)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            self.speaker_vae.load_state_dict(state)
            self.speaker_vae.eval()
            print(f"Loaded Speaker VAE from {vae_path}")

        # 3. Transformer TTS
        tr_path = os.path.join(self.models_dir, "transformer.pt")
        self.transformer_tts = TransformerTTS().to(self.device)
        if os.path.exists(tr_path):
            ckpt = torch.load(tr_path, map_location=self.device, weights_only=False)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            self.transformer_tts.load_state_dict(state)
            self.transformer_tts.eval()
            print(f"Loaded Transformer TTS from {tr_path}")
