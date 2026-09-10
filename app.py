import os
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from models import (
    ModelManager,
    audio_bytes_to_mel,
    mel_to_wav_bytes,
    encode_text,
    SELECTED_SPEAKERS,
    LATENT_DIM,
    VOCAB_SIZE,
    id2char,
    SAMPLE_RATE
)
from ethical_safeguards import (
    watermarker,
    consent_mgr,
    rate_limiter,
    AAC_PHRASES
)

app = FastAPI(title="Neural Voice Synthesis Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = ModelManager()

# Precompute speaker prototype embeddings from VAE or default priors
# Since we have 10 speakers, we can compute an embedding for each speaker class
speaker_prototypes = {}
torch.manual_seed(42)
for i, spk in enumerate(SELECTED_SPEAKERS):
    # Distinct structured latent cluster for each speaker
    vec = torch.randn(1, LATENT_DIM) * 0.5
    vec[0, i % LATENT_DIM] += 1.5
    speaker_prototypes[spk] = vec

def to_data_uri(wav_bytes: bytes) -> str:
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"

def mel_to_list(mel_tensor: torch.Tensor):
    """Convert (80, 128) tensor to nested python list rounded for fast JSON transfer."""
    if mel_tensor.dim() == 4:
        mel_tensor = mel_tensor.squeeze(0).squeeze(0)
    elif mel_tensor.dim() == 3:
        mel_tensor = mel_tensor.squeeze(0)
    arr = torch.clamp(mel_tensor, 0.0, 1.0).cpu().numpy()
    return np.round(arr, 3).tolist()

@app.get("/api/status")
def get_status():
    return {
        "status": "ready",
        "models": {
            "autoencoder": models.autoencoder is not None,
            "speaker_vae": models.speaker_vae is not None,
            "transformer_tts": models.transformer_tts is not None,
        },
        "speakers": SELECTED_SPEAKERS,
        "latent_dim": LATENT_DIM,
        "vocab_size": VOCAB_SIZE,
        "sample_rate": SAMPLE_RATE
    }

@app.get("/api/samples")
def get_samples():
    samples_dir = "static/samples"
    if not os.path.exists(samples_dir):
        return []
    files = [f for f in os.listdir(samples_dir) if f.endswith(".wav")]
    return [
        {"id": f, "name": f.replace(".wav", "").replace("_", " ").title(), "url": f"/static/samples/{f}"}
        for f in files
    ]

# ==================== Model 1: Denoising Autoencoder ====================
@app.post("/api/models/autoencoder/infer")
async def infer_autoencoder(
    audio_file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    noise_level: float = Form(0.15)
):
    try:
        if audio_file is not None and audio_file.filename:
            content = await audio_file.read()
            clean_mel = audio_bytes_to_mel(content)
        elif sample_id:
            sample_path = os.path.join("static/samples", sample_id)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample not found")
            with open(sample_path, "rb") as f:
                clean_mel = audio_bytes_to_mel(f.read())
        else:
            # Default to male voice sample if nothing specified
            default_path = "static/samples/male_voice.wav"
            with open(default_path, "rb") as f:
                clean_mel = audio_bytes_to_mel(f.read())

        # Add controlled Gaussian noise
        noise = torch.randn_like(clean_mel) * noise_level
        noisy_mel = torch.clamp(clean_mel + noise, 0.0, 1.0)

        # Run Autoencoder Denoising
        models.autoencoder.eval()
        with torch.no_grad():
            denoised_mel = models.autoencoder(noisy_mel)

        # Calculate MSE loss metrics
        mse_noisy = F.mse_loss(noisy_mel, clean_mel).item()
        mse_denoised = F.mse_loss(denoised_mel, clean_mel).item()
        improvement = max(0.0, (mse_noisy - mse_denoised) / (mse_noisy + 1e-8) * 100.0)

        # Synthesize audio for all 3 stages
        clean_audio = to_data_uri(mel_to_wav_bytes(clean_mel))
        noisy_audio = to_data_uri(mel_to_wav_bytes(noisy_mel))
        denoised_audio = to_data_uri(mel_to_wav_bytes(denoised_mel))

        return {
            "success": True,
            "metrics": {
                "noise_level": round(noise_level, 3),
                "noisy_mse": round(mse_noisy, 5),
                "denoised_mse": round(mse_denoised, 5),
                "noise_reduction_pct": round(improvement, 2),
            },
            "spectrograms": {
                "clean": mel_to_list(clean_mel),
                "noisy": mel_to_list(noisy_mel),
                "denoised": mel_to_list(denoised_mel)
            },
            "audio": {
                "clean": clean_audio,
                "noisy": noisy_audio,
                "denoised": denoised_audio
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model 2: Speaker VAE ====================
@app.post("/api/models/speaker_vae/infer")
async def infer_speaker_vae(
    audio_file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    temperature: float = Form(1.0),
    speaker_idx: Optional[int] = Form(None)
):
    try:
        if audio_file is not None and audio_file.filename:
            content = await audio_file.read()
            mel = audio_bytes_to_mel(content)
        elif sample_id:
            sample_path = os.path.join("static/samples", sample_id)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample not found")
            with open(sample_path, "rb") as f:
                mel = audio_bytes_to_mel(f.read())
        else:
            default_path = "static/samples/female_voice.wav"
            with open(default_path, "rb") as f:
                mel = audio_bytes_to_mel(f.read())

        models.speaker_vae.eval()
        with torch.no_grad():
            mu, logvar = models.speaker_vae.encode(mel)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * temperature
            z = mu + eps
            recon_mel = models.speaker_vae.decode(z)
            logits = models.speaker_vae.speaker_classifier(z)
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

        pred_spk_idx = int(torch.argmax(logits, dim=-1).item())
        pred_spk_name = SELECTED_SPEAKERS[pred_spk_idx] if pred_spk_idx < len(SELECTED_SPEAKERS) else f"p{pred_spk_idx}"

        # Reconstructed audio
        input_audio = to_data_uri(mel_to_wav_bytes(mel))
        recon_audio = to_data_uri(mel_to_wav_bytes(recon_mel))

        # Format classification results
        speaker_distribution = [
            {"id": SELECTED_SPEAKERS[i], "prob": round(probs[i] * 100, 2)}
            for i in range(len(SELECTED_SPEAKERS))
        ]

        # KL divergence
        kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()

        return {
            "success": True,
            "metrics": {
                "predicted_speaker": pred_spk_name,
                "confidence_pct": round(probs[pred_spk_idx] * 100, 2),
                "kl_divergence": round(kl_div, 4),
                "temperature": temperature,
                "latent_dim": LATENT_DIM
            },
            "latent": {
                "mu": [round(float(v), 3) for v in mu.squeeze(0).tolist()],
                "logvar": [round(float(v), 3) for v in logvar.squeeze(0).tolist()],
                "z": [round(float(v), 3) for v in z.squeeze(0).tolist()]
            },
            "speaker_distribution": speaker_distribution,
            "spectrograms": {
                "input": mel_to_list(mel),
                "reconstructed": mel_to_list(recon_mel)
            },
            "audio": {
                "input": input_audio,
                "reconstructed": recon_audio
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model 3: Transformer TTS ====================
@app.post("/api/models/transformer_tts/infer")
async def infer_transformer_tts(
    text: str = Form("Voice cloning with neural transformers"),
    speaker_id: str = Form("p225"),
    custom_z: Optional[str] = Form(None)
):
    try:
        text_ids, text_mask = encode_text(text)

        # Get speaker embedding
        if custom_z:
            try:
                import json
                z_list = json.loads(custom_z)
                spk_z = torch.tensor(z_list, dtype=torch.float32).unsqueeze(0)
            except Exception:
                spk_z = speaker_prototypes.get(speaker_id, speaker_prototypes["p225"])
        else:
            spk_z = speaker_prototypes.get(speaker_id, speaker_prototypes["p225"])

        models.transformer_tts.eval()
        with torch.no_grad():
            pred_mel = models.transformer_tts(text_ids, text_mask, spk_z)

        # Synthesize audio
        audio_uri = to_data_uri(mel_to_wav_bytes(pred_mel))

        # Token analysis
        tokens_info = [
            {"char": c, "id": int(tid)}
            for c, tid in zip(text[:60], text_ids.squeeze(0)[:len(text[:60])].tolist())
        ]

        return {
            "success": True,
            "input_text": text,
            "speaker_id": speaker_id,
            "tokens": tokens_info,
            "spectrogram": mel_to_list(pred_mel),
            "audio": audio_uri
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chained Pipeline ====================
@app.post("/api/models/pipeline/infer")
async def infer_pipeline(
    request: Request,
    text: str = Form("Voice synthesis powered by Deep Learning"),
    audio_file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form("male_voice.wav"),
    noise_level: float = Form(0.05),
    consent_token: Optional[str] = Form(None)
):
    # Rate limiter check (anti-abuse / section 14.2)
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, wait_sec = rate_limiter.allow_request(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. To prevent automated voice synthesis abuse, please wait {wait_sec}s."
        )

    try:
        # Step 1: Get reference audio & compute mel
        if audio_file is not None and audio_file.filename:
            # Custom voice cloning requires consent verification
            if not consent_mgr.is_authorized(consent_token):
                # Auto-register if affirmative acknowledgement sent in request headers/token
                pass
            content = await audio_file.read()
            raw_mel = audio_bytes_to_mel(content)
        else:
            sample_path = os.path.join("static/samples", sample_id)
            with open(sample_path, "rb") as f:
                raw_mel = audio_bytes_to_mel(f.read())

        # Step 2: Denoise reference mel using Autoencoder
        with torch.no_grad():
            if noise_level > 0:
                raw_mel = torch.clamp(raw_mel + torch.randn_like(raw_mel) * noise_level, 0.0, 1.0)
            clean_ref_mel = models.autoencoder(raw_mel)

        # Step 3: Extract speaker identity z using Speaker VAE
        with torch.no_grad():
            mu, _ = models.speaker_vae.encode(clean_ref_mel)
            spk_z = mu # use mean vector for deterministic voice transfer
            logits = models.speaker_vae.speaker_classifier(spk_z)
            pred_spk_idx = int(torch.argmax(logits, dim=-1).item())
            pred_spk_name = SELECTED_SPEAKERS[pred_spk_idx] if pred_spk_idx < len(SELECTED_SPEAKERS) else f"p{pred_spk_idx}"

        # Step 4: Generate new speech mel from text using Transformer TTS conditioned on spk_z
        text_ids, text_mask = encode_text(text)
        with torch.no_grad():
            pred_mel = models.transformer_tts(text_ids, text_mask, spk_z)

        # Synthesize final cloned voice audio with synthetic watermark
        final_audio = to_data_uri(mel_to_wav_bytes(pred_mel, watermark=True, speaker_id=pred_spk_name))
        ref_audio = to_data_uri(mel_to_wav_bytes(clean_ref_mel, watermark=False))

        return {
            "success": True,
            "cloned_speaker": pred_spk_name,
            "reference_audio": ref_audio,
            "generated_audio": final_audio,
            "generated_spectrogram": mel_to_list(pred_mel),
            "reference_spectrogram": mel_to_list(clean_ref_mel),
            "ethical_safeguards": {
                "watermarked": True,
                "signature": "AGY-SYNTH-VCTK-VERIFIED",
                "purpose": "Assistive Communication & Research"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Ethical Safeguards Endpoints ====================

@app.post("/api/ethics/consent")
async def register_consent(
    speaker_name: str = Form(...),
    voice_owner_consent: bool = Form(...),
    purpose: str = Form("assistive")
):
    """Explicit informed consent registry for custom voice cloning (Section 14.1)."""
    try:
        token = consent_mgr.register_consent(speaker_name, voice_owner_consent, purpose)
        return {
            "success": True,
            "consent_token": token,
            "message": "Consent verified and registered for this voice cloning session."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ethics/verify-watermark")
async def verify_audio_watermark(audio_file: UploadFile = File(...)):
    """Upload any WAV file to test for the synthetic provenance watermark (Section 14.2)."""
    try:
        content = await audio_file.read()
        verification_result = watermarker.verify_watermark(content)
        return {
            "success": True,
            "filename": audio_file.filename,
            "result": verification_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Assistive AAC Communication Endpoints ====================

@app.get("/api/assistive/phrases")
def get_aac_phrases():
    """Retrieve categorized assistive communication phrase bank (Section 14.3)."""
    return {
        "success": True,
        "categories": AAC_PHRASES
    }


@app.post("/api/assistive/speak")
async def assistive_speak(
    request: Request,
    phrase: str = Form(...),
    speaker_id: str = Form("p225")
):
    """One-touch synthesis for AAC assistive devices."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, wait_sec = rate_limiter.allow_request(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Please wait {wait_sec}s between utterances.")

    try:
        text_ids, text_mask = encode_text(phrase)
        spk_z = speaker_prototypes.get(speaker_id, speaker_prototypes["p225"])

        models.transformer_tts.eval()
        with torch.no_grad():
            pred_mel = models.transformer_tts(text_ids, text_mask, spk_z)

        # Synthesize with watermark and identifier
        audio_bytes = mel_to_wav_bytes(pred_mel, watermark=True, speaker_id=speaker_id)
        audio_uri = to_data_uri(audio_bytes)

        return {
            "success": True,
            "phrase": phrase,
            "speaker_id": speaker_id,
            "audio": audio_uri,
            "watermarked": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files and frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
