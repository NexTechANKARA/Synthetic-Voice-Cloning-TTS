"""The 5 core models - architectures are IDENTICAL to the original notebooks.
Only num_speakers / vocab_size are constructor arguments (they depend on the dataset)."""
import math
import torch
import torch.nn as nn
from .common import N_MELS, SEG, LATENT_DIM, PAD_ID


# ============ Model 1: Denoising Autoencoder ============
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


# ============ Model 2: Speaker VAE ============
class SpeakerVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_speakers=10):
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 10, 16)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.speaker_classifier(z)


# ============ Model 3: Transformer (text -> coarse mel) ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTTS(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, enc_layers=3, dec_layers=3,
                 n_mels=N_MELS, seg_frames=SEG, speaker_dim=LATENT_DIM):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers, enable_nested_tensor=False)
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
        enc_out = enc_out + self.speaker_proj(speaker_emb).unsqueeze(1)
        queries = self.mel_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.mel_decoder(tgt=queries, memory=enc_out, memory_key_padding_mask=key_padding_mask)
        mel_out = torch.sigmoid(self.output_proj(dec_out))
        return mel_out.transpose(1, 2).unsqueeze(1)      # (B, 1, n_mels, seg_frames)


# ============ Model 3 (upgraded): FastSpeech-style Transformer ============
class ConvBlock(nn.Module):
    """Conv1d -> ReLU -> LayerNorm -> Dropout, on (B, C, T)."""
    def __init__(self, cin, cout, k=3, p=0.1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.norm = nn.LayerNorm(cout)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.drop(self.norm(x.transpose(1, 2)).transpose(1, 2))


class DurationPredictor(nn.Module):
    """Predicts log(1 + frames) for every character."""
    def __init__(self, d_model=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(d_model, hidden), ConvBlock(hidden, hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, h, text_mask):                       # h: (B, L, D)
        x = self.net(h.transpose(1, 2)).transpose(1, 2)
        return self.out(x).squeeze(-1) * text_mask           # (B, L)


class AlignmentModule(nn.Module):
    """Scores every (mel frame, character) pair. Used only during training to learn durations
    (forward-sum loss + monotonic alignment search, as in RAD-TTS / Glow-TTS)."""
    def __init__(self, d_model=256, n_mels=N_MELS, att_dim=80, temperature=0.1):
        super().__init__()
        self.key = nn.Sequential(nn.Conv1d(d_model, 2 * d_model, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(2 * d_model, att_dim, 1))
        self.query = nn.Sequential(nn.Conv1d(n_mels, 2 * n_mels, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(2 * n_mels, n_mels, 1), nn.ReLU(),
                                   nn.Conv1d(n_mels, att_dim, 1))
        self.temperature = temperature

    def forward(self, h, mel, text_mask, log_prior=None):
        """h: (B, L, D), mel: (B, 80, T). Returns log-probabilities over characters, (B, T, L)."""
        k = self.key(h.transpose(1, 2))                        # (B, A, L)
        q = self.query(mel)                                    # (B, A, T)
        dist = ((q.unsqueeze(-1) - k.unsqueeze(2)) ** 2).sum(1)  # (B, T, L)
        logits = -self.temperature * dist
        if log_prior is not None:
            logits = logits + log_prior
        logits = logits.masked_fill(text_mask.unsqueeze(1) == 0, -1e4)
        return torch.log_softmax(logits, dim=-1)


def length_regulate(h, durations, max_len=None):
    """Repeat each character encoding by its duration. h: (B, L, D), durations: (B, L) ints.
    Returns (B, T, D) and the frame lengths (B,)."""
    out = [torch.repeat_interleave(h[b], durations[b].clamp(min=0), dim=0) for b in range(h.size(0))]
    lengths = torch.tensor([o.size(0) for o in out], device=h.device)
    T = max_len or max(int(lengths.max()), 1)
    padded = h.new_zeros(h.size(0), T, h.size(2))
    for b, o in enumerate(out):
        n = min(o.size(0), T)
        padded[b, :n] = o[:n]
    return padded, lengths.clamp(max=T)


class FastSpeechTTS(nn.Module):
    """Text -> mel with explicit durations. Encoder layers match TransformerTTS, so the encoder can
    start from the baseline's trained weights."""
    def __init__(self, vocab_size, d_model=256, nhead=4, enc_layers=3, dec_layers=4,
                 n_mels=N_MELS, speaker_dim=LATENT_DIM):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers, enable_nested_tensor=False)
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        self.duration = DurationPredictor(d_model)
        self.aligner = AlignmentModule(d_model, n_mels)
        self.dec_pos = PositionalEncoding(d_model, max_len=4096)
        self.dec_speaker = nn.Linear(speaker_dim, d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, batch_first=True)
        self.mel_decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers, enable_nested_tensor=False)
        self.output_proj = nn.Linear(d_model, n_mels)

    def encode(self, text_ids, text_mask, speaker_emb):
        x = self.pos_enc(self.text_embed(text_ids))
        h = self.text_encoder(x, src_key_padding_mask=(text_mask == 0))
        return (h + self.speaker_proj(speaker_emb).unsqueeze(1)) * text_mask.unsqueeze(-1)

    def align(self, text_ids, text_mask, mel, log_prior=None):
        """Alignment scores from the raw character embeddings (not the encoder output), so characters are
        always distinguishable to the aligner even while the encoder is still learning (as in RAD-TTS)."""
        e = self.text_embed(text_ids) * text_mask.unsqueeze(-1)
        return self.aligner(e, mel, text_mask, log_prior)

    def decode(self, h, durations, speaker_emb, max_len=None):
        x, lengths = length_regulate(h, durations, max_len)
        T = x.size(1)
        frame_mask = torch.arange(T, device=x.device)[None] < lengths[:, None]   # (B, T)
        x = self.dec_pos(x) + self.dec_speaker(speaker_emb).unsqueeze(1)
        y = self.mel_decoder(x, src_key_padding_mask=~frame_mask)
        mel = torch.sigmoid(self.output_proj(y)) * frame_mask.unsqueeze(-1)
        return mel.transpose(1, 2), frame_mask                                    # (B, 80, T)

    def predict_durations(self, h, text_mask, pace=1.0):
        logd = self.duration(h.detach(), text_mask)
        d = torch.round((torch.exp(logd) - 1).clamp(min=0) * pace).long().clamp(min=1)
        return d * text_mask.long()

    @torch.no_grad()
    def infer(self, text_ids, text_mask, speaker_emb, pace=1.0, max_frames=1024):
        """Returns mel (B, 1, 80, T) and frame lengths (B,)."""
        h = self.encode(text_ids, text_mask, speaker_emb)
        d = self.predict_durations(h, text_mask, pace)
        T = min(int(d.sum(1).max()), max_frames)
        mel, frame_mask = self.decode(h, d, speaker_emb, max_len=T)
        return mel.unsqueeze(1), frame_mask.sum(1)


# ============ Model 4: Diffusion denoiser ============
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionDenoiser(nn.Module):
    def __init__(self, time_dim=128, speaker_dim=LATENT_DIM):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_dim),
                                      nn.Linear(time_dim, time_dim), nn.ReLU())
        self.speaker_proj = nn.Linear(speaker_dim, time_dim)
        self.down1 = nn.Conv2d(2, 32, 3, padding=1)
        self.down2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.film1 = nn.Linear(time_dim, 32 * 2)
        self.film2 = nn.Linear(time_dim, 64 * 2)
        self.film3 = nn.Linear(time_dim, 128 * 2)
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(64 + 64, 32, 3, stride=2, padding=1, output_padding=1)
        self.out_conv = nn.Conv2d(32 + 32, 1, 3, padding=1)
        self.act = nn.ReLU()

    @staticmethod
    def film(h, params):
        scale, shift = params.chunk(2, dim=-1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, noisy_mel, t, coarse_mel, speaker_emb):
        temb = self.time_mlp(t) + self.speaker_proj(speaker_emb)
        x = torch.cat([noisy_mel, coarse_mel], dim=1)
        h1 = self.act(self.film(self.down1(x), self.film1(temb)))
        h2 = self.act(self.film(self.down2(h1), self.film2(temb)))
        h3 = self.act(self.film(self.down3(h2), self.film3(temb)))
        u1 = torch.cat([self.act(self.up1(h3)), h2], dim=1)
        u2 = torch.cat([self.act(self.up2(u1)), h1], dim=1)
        return self.out_conv(u2)


# ============ Model 5: GAN vocoder ============
class Generator(nn.Module):
    """Mel (B,1,80,128) -> waveform (B,1,32768)."""
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        self.input_conv = nn.Conv1d(n_mels, 256, kernel_size=7, padding=3)
        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 8, 4, 2),
            nn.ConvTranspose1d(128, 64, 8, 4, 2),
            nn.ConvTranspose1d(64, 32, 8, 4, 2),
            nn.ConvTranspose1d(32, 16, 8, 4, 2),
        ])
        self.res_blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=1) for ch in [128, 64, 32, 16]])
        self.output_conv = nn.Conv1d(16, 1, 7, padding=3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, mel):
        x = self.act(self.input_conv(mel.squeeze(1)))
        for up, res in zip(self.up_blocks, self.res_blocks):
            x = self.act(up(x))
            x = x + self.act(res(x))
        return torch.tanh(self.output_conv(x))


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 16, 15, 1, 7),
            nn.Conv1d(16, 64, 41, 4, 20, groups=4),
            nn.Conv1d(64, 256, 41, 4, 20, groups=16),
            nn.Conv1d(256, 512, 41, 4, 20, groups=16),
            nn.Conv1d(512, 512, 5, 1, 2),
        ])
        self.out = nn.Conv1d(512, 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = self.act(layer(x))
            feats.append(x)
        return self.out(x), feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.pool = nn.AvgPool1d(4, 2, 2)

    def forward(self, x):
        outs, feats = [], []
        for i, d in enumerate(self.discs):
            if i > 0:
                x = self.pool(x)
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        return outs, feats
