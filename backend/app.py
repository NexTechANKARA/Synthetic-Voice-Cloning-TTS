"""Voice Cloner API: the five trained models behind three endpoints.

Run from the repository folder:  uvicorn backend.app:app --port 7860
Settings (environment variables):
  SUPABASE_URL, SUPABASE_KEY  the Supabase project and its publishable key, used to check sign-in tokens
  ALLOWED_ORIGINS             comma-separated sites allowed to call the API (the Netlify site)
  VC_ROOT                     folder holding vc_lib/, data/ and checkpoints/ (default: the repository)
  VC_DEV_NO_AUTH=1            skip the sign-in check, for local testing only
"""
import io
import os
import sys
import time
import json
import hashlib
import logging
import tempfile
import threading
from collections import defaultdict, deque
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("VC_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT))
from vc_lib.common import SR, clean_text  # noqa: E402

log = logging.getLogger("voice-cloner")
logging.basicConfig(level=logging.INFO)

MAX_CHARS = 1000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SPEECH_S, MAX_CLIP_S = 3.0, 60.0
LIMITS = {"speak": (30, 3600), "embed": (10, 3600)}      # requests per window (seconds), per user
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
NO_AUTH = os.environ.get("VC_DEV_NO_AUTH") == "1"
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

state = {"pipe": None, "error": None, "started": time.time()}
model_lock = threading.Lock()


def load_models():
    try:
        from vc_lib.infer import VoicePipeline
        t0 = time.time()
        state["pipe"] = VoicePipeline(str(ROOT), diffusion_steps=50, cfg=1.0, strength=0.1)
        log.info("Models loaded in %.1f s", time.time() - t0)
    except Exception as e:                                   # reported by /health instead of crashing
        state["error"] = f"{type(e).__name__}: {e}"
        log.exception("Model loading failed")


class BodyLimit:
    """Stops reading a request body once it passes `limit` bytes, before FastAPI parses it.
    Checks the Content-Length header and also counts the bytes actually received."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        too_big = HTTPException(413, "That file is over 20 MB. Use a clip under one minute.")
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.limit:
            return await JSONResponse({"error": too_big.detail}, status_code=413)(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.limit:
                raise too_big                                      # FastAPI passes HTTPException through unchanged
            return message

        await self.app(scope, counted, send)


app = FastAPI(title="Voice Cloner API", docs_url=None, redoc_url=None)
app.add_middleware(BodyLimit, limit=MAX_UPLOAD_BYTES + 1024 * 1024)   # 1 MB headroom for the form fields
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()],
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
                   expose_headers=["X-Audio-Seconds"])
threading.Thread(target=load_models, daemon=True).start()


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, exc: RequestValidationError):
    return JSONResponse({"error": "The request was not in the expected format."}, status_code=422)


def pipeline():
    if state["error"]:
        raise HTTPException(503, "The voice models failed to load. Try again later.")
    if state["pipe"] is None:
        raise HTTPException(503, "The voice server is starting up. Try again in a few seconds.", {"Retry-After": "5"})
    return state["pipe"]


# ---------------------------------------------------------------- sign-in and limits
_tokens: dict[str, tuple[str, float]] = {}                  # token hash -> (user id, checked at)
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_limit_lock = threading.Lock()


def current_user(authorization: str | None) -> str:
    """Return the Supabase user id for a bearer token (checked with Supabase, cached for 5 minutes)."""
    if NO_AUTH:
        return "dev"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in to use the voice server.")
    token = authorization[7:].strip()
    key = hashlib.sha256(token.encode()).hexdigest()
    hit = _tokens.get(key)
    if hit and time.time() - hit[1] < 300:
        return hit[0]
    if not SUPABASE_URL:
        raise HTTPException(500, "The server is missing its sign-in settings.")
    try:
        r = httpx.get(f"{SUPABASE_URL}/auth/v1/user", headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_KEY}, timeout=10)
    except httpx.HTTPError:
        raise HTTPException(503, "Couldn't check your sign-in. Try again in a moment.")
    if r.status_code != 200 or not r.json().get("id"):
        raise HTTPException(401, "Your sign-in has expired. Sign in again.")
    uid = r.json()["id"]
    if len(_tokens) > 10000:
        _tokens.clear()
    _tokens[key] = (uid, time.time())
    return uid


def check_limit(uid: str, kind: str):
    n, window = LIMITS[kind]
    now = time.time()
    with _limit_lock:
        q = _hits[(uid, kind)]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= n:
            wait = int(window - (now - q[0])) + 1
            raise HTTPException(429, f"You've reached {n} per hour. Try again in {max(1, wait // 60)} minutes.", {"Retry-After": str(wait)})
        q.append(now)


# ---------------------------------------------------------------- built-in voices
def voice_info():
    info_file = ROOT / "data" / "speaker_info.json"          # optional: {"p225": {"gender": "female", "accent": "English", "region": "Southern England"}}
    return json.loads(info_file.read_text()) if info_file.exists() else {}


@app.get("/health")
def health():
    return {"ready": state["pipe"] is not None, "error": state["error"] is not None,
            "uptime_s": round(time.time() - state["started"])}


@app.get("/voices")
def voices():
    meta = pipeline().meta
    info = voice_info()
    out = [{"id": s, "seen_in_training": True, **info.get(s, {})} for s in meta["train_speakers"]]
    out += [{"id": s, "seen_in_training": False, **info.get(s, {})} for s in meta["holdout"]]
    return {"voices": out}


# ---------------------------------------------------------------- voice from a clip
@app.post("/embed")
def embed(file: UploadFile = File(...), authorization: str | None = Header(None)):
    uid = current_user(authorization)
    pipe = pipeline()
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 20 MB. Use a clip under one minute.")
    import librosa
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "clip"
        raw.write_bytes(data)
        try:
            y, _ = librosa.load(str(raw), sr=SR, mono=True, duration=MAX_CLIP_S)
        except Exception:
            raise HTTPException(400, "Couldn't read that audio. Upload a WAV, MP3, M4A or OGG file, or record in the browser.")
        if y.size == 0 or not np.isfinite(y).all():
            raise HTTPException(400, "The clip is empty.")
        if np.abs(y).max() < 0.01:                                # silence: split() would count it all as speech
            raise HTTPException(400, "The clip is silent. Check the microphone, then record again.")
        speech = sum(int(b - a) for a, b in librosa.effects.split(y, top_db=30)) / SR
        if speech < MIN_SPEECH_S:
            raise HTTPException(400, f"The clip has only {speech:.1f} seconds of speech. Record at least 5 seconds in a quiet room.")
        check_limit(uid, "embed")                            # count only clips that reach the model
        clean = Path(tmp) / "clip.wav"
        sf.write(str(clean), y, SR, subtype="PCM_16")
        with model_lock:
            emb = pipe.embed_reference([str(clean)]).detach().float().cpu()
    peak = float(np.abs(y).max())
    return {"embedding": [round(float(v), 6) for v in emb], "speech_seconds": round(speech, 1),
            "checks": {"long_enough": speech >= 5.0, "not_clipped": peak < 0.99}}


# ---------------------------------------------------------------- speak
class SpeakRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS * 2)
    voice: str | None = None                                  # a built-in voice id, e.g. "p225"
    embedding: list[float] | None = Field(None, min_length=64, max_length=64)   # or a voice from /embed
    speed: float = 1.0                                        # 0.75 to 1.33, higher is faster


@app.post("/speak")
def speak(req: SpeakRequest, authorization: str | None = Header(None)):
    uid = current_user(authorization)
    pipe = pipeline()
    text = req.text.strip()
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"Keep the text under {MAX_CHARS} characters.")
    if not clean_text(text):
        raise HTTPException(400, "There are no words to speak. Type English words; digits are read as numbers.")
    speaker, emb = None, None
    if req.embedding is not None:
        e = torch.tensor(req.embedding, dtype=torch.float32)
        if not torch.isfinite(e).all() or e.abs().max() > 100:
            raise HTTPException(400, "That voice is not valid. Add your voice again.")
        emb = e
    elif req.voice in pipe.speaker_mean:
        speaker = req.voice
    else:
        raise HTTPException(400, "Choose a voice first.")
    check_limit(uid, "speak")
    speed = min(max(req.speed, 0.75), 1.33)
    with model_lock:
        y = pipe.synthesize(text, speaker=speaker, speaker_emb=emb, pace=1.0 / speed)
    if y.size == 0:
        raise HTTPException(400, "There are no words to speak. Type English words; digits are read as numbers.")
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{len(y) / SR:.2f}", "Cache-Control": "no-store"})
