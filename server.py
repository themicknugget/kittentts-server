import io
import os
import logging
import onnxruntime as ort
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("KITTENTTS_MODEL", "KittenML/kitten-tts-mini-0.8")
SAMPLE_RATE = 24000

# ORT_PROVIDERS: comma-separated list, e.g. "VulkanExecutionProvider,CPUExecutionProvider"
# If unset, ORT picks from get_available_providers() automatically.
_providers_env = os.environ.get("ORT_PROVIDERS", "").strip()
REQUESTED_PROVIDERS = [p.strip() for p in _providers_env.split(",") if p.strip()] or None

# Monkey-patch ort.InferenceSession so KittenTTS (which doesn't accept a providers arg)
# uses our provider list when it constructs its session.
_OrigInferenceSession = ort.InferenceSession

class _PatchedSession(_OrigInferenceSession):
    def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        effective = providers if providers is not None else REQUESTED_PROVIDERS
        super().__init__(path_or_bytes, sess_options=sess_options, providers=effective, **kwargs)

ort.InferenceSession = _PatchedSession

tts = None
active_providers: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, active_providers
    available = ort.get_available_providers()
    logger.info(f"ORT available providers: {available}")
    logger.info(f"ORT requested providers: {REQUESTED_PROVIDERS or '(auto)'}")
    logger.info(f"Loading KittenTTS model: {MODEL_ID}")
    from kittentts import KittenTTS
    tts = KittenTTS(MODEL_ID)
    active_providers = tts.model.session.get_providers()
    logger.info(f"ORT active providers: {active_providers}")
    yield


app = FastAPI(title="KittenTTS OpenAI-compatible server", lifespan=lifespan)

# Map OpenAI voice aliases to KittenTTS mini-0.8 voices (expr-voice-{2-5}-{m,f})
VOICE_MAP = {
    "alloy":   "expr-voice-2-m",
    "echo":    "expr-voice-3-m",
    "fable":   "expr-voice-2-f",
    "onyx":    "expr-voice-4-m",
    "nova":    "expr-voice-3-f",
    "shimmer": "expr-voice-4-f",
}

NATIVE_VOICES = [
    "expr-voice-2-m", "expr-voice-2-f",
    "expr-voice-3-m", "expr-voice-3-f",
    "expr-voice-4-m", "expr-voice-4-f",
    "expr-voice-5-m", "expr-voice-5-f",
]


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    speed: float = 1.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "ready": tts is not None,
        "ort_providers": active_providers,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model"},
            {"id": "tts-1-hd", "object": "model"},
        ],
    }


@app.get("/v1/audio/voices")
def list_voices():
    return {"voices": NATIVE_VOICES}


@app.post("/v1/audio/speech")
def audio_speech(req: SpeechRequest):
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice = VOICE_MAP.get(req.voice.lower(), req.voice)
    speed = max(0.25, min(4.0, req.speed))

    try:
        audio = tts.generate(req.input, voice=voice, speed=speed)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Ensure 1D float32
    if audio.ndim > 1:
        audio = audio.squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("KITTENTTS_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
