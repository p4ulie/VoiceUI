"""VoiceUI backend: serves the UI and proxies STT (faster-whisper), chat
(llama.cpp), and TTS (Piper). One process, models held in memory.

ponytail: single global model instance, no lock — single-user local.
Add a queue/lock if concurrent users ever matter.
"""
import io
import os
import wave
from functools import cache

import httpx
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
PIPER_VOICE = os.getenv("PIPER_VOICE", "voices/en_US-lessac-medium.onnx")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@cache
def stt_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)


@cache
def tts_voice():
    from piper import PiperVoice
    return PiperVoice.load(os.path.join(HERE, PIPER_VOICE))


def synth_wav(text: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        tts_voice().synthesize_wav(text, wf)
    return buf.getvalue()


def transcribe(audio: bytes) -> str:
    # faster-whisper decodes webm/opus/wav via bundled PyAV — no system ffmpeg needed.
    segments, _ = stt_model().transcribe(io.BytesIO(audio), beam_size=1)
    return "".join(s.text for s in segments).strip()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.post("/stt")
async def stt(file: UploadFile):
    return {"text": transcribe(await file.read())}


@app.get("/models")
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{LLAMA_URL}/models")
    return [m["id"] for m in r.json().get("data", [])]


class ChatReq(BaseModel):
    model: str
    messages: list[dict]


@app.post("/chat")
async def chat(req: ChatReq):
    messages = req.messages
    if SYSTEM_PROMPT:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    payload = {"model": req.model, "messages": messages, "stream": True}

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{LLAMA_URL}/chat/completions", json=payload
            ) as r:
                async for chunk in r.aiter_raw():
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


class TTSReq(BaseModel):
    text: str


@app.post("/tts")
def tts(req: TTSReq):
    return Response(synth_wav(req.text), media_type="audio/wav")


def selftest():
    """TTS 'hello world' -> STT round-trip. Fails if either model path is broken."""
    wav = synth_wav("hello world")
    assert wav[:4] == b"RIFF", "piper did not produce a WAV"
    text = transcribe(wav).lower()
    print(f"round-trip transcript: {text!r}")
    assert "hello" in text or "world" in text, f"STT did not recover text: {text!r}"
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
