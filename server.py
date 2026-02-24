import io
import os
import logging
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

tts = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    logger.info(f"Loading KittenTTS model: {MODEL_ID}")
    from kittentts import KittenTTS
    tts = KittenTTS(model=MODEL_ID)
    logger.info("Model ready")
    yield


app = FastAPI(title="KittenTTS OpenAI-compatible server", lifespan=lifespan)

# Map OpenAI voice aliases to KittenTTS voices
VOICE_MAP = {
    "alloy":   "Jasper",
    "echo":    "Bruno",
    "fable":   "Bella",
    "onyx":    "Hugo",
    "nova":    "Luna",
    "shimmer": "Rosie",
    # Native KittenTTS voices pass through
    "bella":  "Bella",
    "jasper": "Jasper",
    "luna":   "Luna",
    "bruno":  "Bruno",
    "rosie":  "Rosie",
    "hugo":   "Hugo",
    "kiki":   "Kiki",
    "leo":    "Leo",
}

NATIVE_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "ready": tts is not None}


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
