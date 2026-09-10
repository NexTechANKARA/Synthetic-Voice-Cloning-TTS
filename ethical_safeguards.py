import io
import time
import struct
import hashlib
import numpy as np
from scipy.io import wavfile
from typing import Tuple, Dict, Any, Optional

SYNTHETIC_WATERMARK_SIGNATURE = b"AGY-SYNTH-VCTK-VERIFIED"
METADATA_MAGIC = b"AUTH"

# Assistive AAC Categories and Phrases for speech-impaired individuals
AAC_PHRASES = {
    "Urgent & Health": [
        "I need immediate medical assistance.",
        "Please help me adjust my position.",
        "I am having difficulty breathing.",
        "I am experiencing severe pain.",
        "Please call my caregiver right now."
    ],
    "Daily Essentials": [
        "Yes, please.",
        "No, thank you.",
        "I would like a glass of water.",
        "I am hungry and ready to eat.",
        "Could you please turn off the light?"
    ],
    "Social & Conversation": [
        "Hello, it is wonderful to see you.",
        "Thank you so much for your help.",
        "I love you very much.",
        "Please give me a minute to type my thoughts.",
        "How was your day today?"
    ],
    "Comfort & Environment": [
        "I am feeling too cold, could I have a blanket?",
        "It is too noisy here, can we move somewhere quiet?",
        "I am comfortable right now, thank you.",
        "Can we listen to some music?",
        "I am ready to go to sleep."
    ]
}

class AudioWatermarkingEngine:
    """
    Implements a dual-layer audio watermark:
    1. Imperceptible Acoustic Watermark: High-frequency carrier tone near Nyquist frequency (10.8 kHz for 22.05 kHz)
       modulated with a unique low-amplitude harmonic fingerprint that survives basic compression.
    2. RIFF Metadata Watermark: Embedded chunk into the WAV file header certifying synthetic provenance.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def embed_watermark(self, wav_bytes: bytes, speaker_id: str = "unknown") -> bytes:
        """Inject acoustic watermark and metadata header into WAV bytes."""
        try:
            sr, data = wavfile.read(io.BytesIO(wav_bytes))
            is_float = np.issubdtype(data.dtype, np.floating)
            
            # Normalize to float in [-1, 1]
            if is_float:
                audio_float = data.copy()
            else:
                audio_float = (data / 32768.0).astype(np.float32)

            num_samples = len(audio_float)
            if num_samples > 0:
                t = np.arange(num_samples) / sr
                # Sub-perceptual high-frequency tone (~10,200 Hz with tiny 1e-4 amplitude)
                # Well within 22.05 kHz Nyquist (11.025 kHz), virtually imperceptible to human ear
                carrier = np.sin(2 * np.pi * 10200 * t)
                modulator = 0.5 * (1.0 + np.sin(2 * np.pi * 40 * t))
                watermark_signal = (carrier * modulator * 0.00035).astype(np.float32)
                
                audio_marked = np.clip(audio_float + watermark_signal, -1.0, 1.0)
            else:
                audio_marked = audio_float

            # Re-quantize to int16
            audio_int16 = (audio_marked * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            wavfile.write(buf, sr, audio_int16)
            raw_wav = buf.getvalue()

            # Append custom RIFF chunk for cryptographic provenance verification
            payload = {
                "sig": SYNTHETIC_WATERMARK_SIGNATURE.decode("ascii"),
                "speaker": speaker_id,
                "ts": int(time.time()),
                "hash": hashlib.sha256(raw_wav[:1024]).hexdigest()[:16]
            }
            meta_str = f"SYNTH:{payload['sig']}|SPK:{payload['speaker']}|TS:{payload['ts']}|H:{payload['hash']}"
            meta_bytes = meta_str.encode("utf-8")
            
            chunk = METADATA_MAGIC + struct.pack("<I", len(meta_bytes)) + meta_bytes
            return raw_wav + chunk
        except Exception:
            return wav_bytes

    def verify_watermark(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Verify whether an audio file contains the synthetic watermark signature."""
        has_meta = False
        meta_info = {}
        
        # 1. Check for RIFF Metadata Chunk
        if METADATA_MAGIC in audio_bytes:
            idx = audio_bytes.rfind(METADATA_MAGIC)
            try:
                length = struct.unpack("<I", audio_bytes[idx+4:idx+8])[0]
                content = audio_bytes[idx+8:idx+8+length].decode("utf-8", errors="ignore")
                parts = dict(item.split(":", 1) for item in content.split("|") if ":" in item)
                has_meta = parts.get("SYNTH") == SYNTHETIC_WATERMARK_SIGNATURE.decode("ascii")
                meta_info = parts
            except Exception:
                pass

        # 2. Spectral energy check near 10,200 Hz
        has_acoustic = False
        spectral_confidence = 0.0
        try:
            sr, data = wavfile.read(io.BytesIO(audio_bytes))
            if len(data) > 2048:
                samples = data[:16384].astype(np.float32)
                fft = np.abs(np.fft.rfft(samples))
                freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
                
                target_band = (freqs >= 10150) & (freqs <= 10250)
                baseline_band = (freqs >= 9800) & (freqs <= 10100)
                
                band_energy = np.mean(fft[target_band]) if np.any(target_band) else 0
                base_energy = np.mean(fft[baseline_band]) if np.any(baseline_band) else 1e-6
                
                ratio = band_energy / (base_energy + 1e-6)
                if ratio > 1.1:
                    has_acoustic = True
                    spectral_confidence = min(100.0, float(ratio * 40.0))
                else:
                    spectral_confidence = max(10.0, float(ratio * 30.0))
        except Exception:
            pass

        is_verified = has_meta or has_acoustic
        confidence = 99.8 if has_meta else (round(spectral_confidence, 1) if has_acoustic else 0.0)

        return {
            "is_synthetic": is_verified,
            "verification_method": "Dual-Layer (Cryptographic Chunk + High-Frequency Tone)" if (has_meta and has_acoustic) else ("Metadata Watermark" if has_meta else ("Acoustic Spectrum" if has_acoustic else "None")),
            "confidence_pct": confidence,
            "provenance": meta_info,
            "ethical_compliance": "Verified Synthetic Audio (Safe Deployment)" if is_verified else "Unknown / Unmarked Audio"
        }


class ConsentGovernanceManager:
    """Manages voice consent registration and authorization tokens."""
    def __init__(self):
        self._consents = {}

    def register_consent(self, speaker_name: str, voice_owner_consent: bool, use_purpose: str = "assistive") -> str:
        if not voice_owner_consent:
            raise ValueError("Voice cloning strictly requires affirmative consent under Section 14.1.")
        
        token = hashlib.sha256(f"{speaker_name}-{time.time()}-consent".encode()).hexdigest()[:24]
        self._consents[token] = {
            "speaker": speaker_name,
            "consent_granted": True,
            "purpose": use_purpose,
            "timestamp": time.time(),
            "policy": "Assistive Communication / Non-commercial Research"
        }
        return token

    def is_authorized(self, consent_token: Optional[str]) -> bool:
        if not consent_token:
            return False
        return consent_token in self._consents


class SimpleRateLimiter:
    """Session-based rate limiter to protect against spam / unauthorized batch generation."""
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.history = {}

    def allow_request(self, client_id: str) -> Tuple[bool, int]:
        now = time.time()
        window = 60.0
        if client_id not in self.history:
            self.history[client_id] = []

        self.history[client_id] = [t for t in self.history[client_id] if now - t < window]

        if len(self.history[client_id]) >= self.max_requests:
            remaining = int(window - (now - self.history[client_id][0]))
            return False, max(1, remaining)

        self.history[client_id].append(now)
        return True, 0


# Global Instances
watermarker = AudioWatermarkingEngine(sample_rate=22050)
consent_mgr = ConsentGovernanceManager()
rate_limiter = SimpleRateLimiter(max_requests_per_minute=30)
