"""VoiceUI backend: serves the UI and proxies STT (faster-whisper), chat
(OpenAI-compatible LLM server), and TTS (Piper). One process, models held in
memory, settings persisted to settings.json.

ponytail: single global model instance, no lock — single-user local.
Add a queue/lock if concurrent users ever matter.
"""
import glob
import io
import json
import os
import time
import wave

import httpx
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(HERE, "voices")
SETTINGS_FILE = os.path.join(HERE, "settings.json")

DEFAULTS = {
    "llm_url": os.getenv("LLAMA_URL", "http://localhost:8080/v1"),
    "api_token": os.getenv("API_TOKEN", ""),
    "model": os.getenv("MODEL", ""),
    "whisper_model": os.getenv("WHISPER_MODEL", "small"),
    "whisper_compute": os.getenv("WHISPER_COMPUTE", "int8"),
    "piper_voice": os.getenv("PIPER_VOICE", "en_US-lessac-medium.onnx"),
    "wake_word": os.getenv("WAKE_WORD", "computer"),
    "stop_word": os.getenv("STOP_WORD", "stop"),
    "wake_timeout": int(os.getenv("WAKE_TIMEOUT", "30")),
    "auto_tts": True,
    "system_prompt": os.getenv("SYSTEM_PROMPT", ""),
}


def load_settings():
    s = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        s.update(json.load(open(SETTINGS_FILE)))
    return s


SETTINGS = load_settings()

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


# --- models loaded lazily, reloaded when the relevant setting changes ---
_stt = {}
_tts = {}


def stt_model():
    key = (SETTINGS["whisper_model"], SETTINGS["whisper_compute"])
    if _stt.get("key") != key:
        from faster_whisper import WhisperModel
        _stt["m"] = WhisperModel(key[0], device="cpu", compute_type=key[1])
        _stt["key"] = key
    return _stt["m"]


def tts_voice():
    key = SETTINGS["piper_voice"]
    if _tts.get("key") != key:
        from piper import PiperVoice
        _tts["m"] = PiperVoice.load(os.path.join(VOICES_DIR, key))
        _tts["key"] = key
    return _tts["m"]


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


# --- settings ---
@app.get("/settings")
def get_settings():
    return {**SETTINGS, "api_token": "***" if SETTINGS["api_token"] else ""}


@app.post("/settings")
def post_settings(patch: dict):
    # keep existing token if the UI sent the masked placeholder
    if patch.get("api_token") == "***":
        patch.pop("api_token")
    SETTINGS.update({k: v for k, v in patch.items() if k in DEFAULTS})
    json.dump(SETTINGS, open(SETTINGS_FILE, "w"), indent=2)
    return get_settings()


@app.get("/voices")
def voices():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(VOICES_DIR, "*.onnx")))


# --- LLM proxy ---
def _auth_headers():
    t = SETTINGS["api_token"]
    return {"Authorization": f"Bearer {t}"} if t else {}


@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{SETTINGS['llm_url']}/models", headers=_auth_headers())
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


@app.post("/test_llm")
async def test_llm(body: dict):
    """Check the configured LLM URL is reachable and the model actually answers."""
    url, model = SETTINGS["llm_url"], body.get("model", "")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{url}/models", headers=_auth_headers())
            if r.status_code != 200:
                return {"ok": False, "detail": f"connect: HTTP {r.status_code} {r.text[:120]}"}
            if not model:
                return {"ok": True, "detail": "reachable (no model selected to test)"}
            r = await client.post(
                f"{url}/chat/completions", headers=_auth_headers(),
                json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1, "stream": False},
            )
        if r.status_code != 200:
            return {"ok": False, "detail": f"{model}: HTTP {r.status_code} {r.text[:120]}"}
        if not r.json().get("choices"):
            return {"ok": False, "detail": f"{model}: no completion returned"}
        return {"ok": True, "detail": f"{model} responded"}
    except Exception as e:
        return {"ok": False, "detail": f"connect failed: {e}"}


class ChatReq(BaseModel):
    model: str
    messages: list[dict]


@app.post("/chat")
async def chat(req: ChatReq):
    messages = req.messages
    if SETTINGS["system_prompt"]:
        messages = [{"role": "system", "content": SETTINGS["system_prompt"]}, *messages]
    payload = {"model": req.model, "messages": messages, "stream": True}

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{SETTINGS['llm_url']}/chat/completions",
                json=payload, headers=_auth_headers(),
            ) as r:
                async for chunk in r.aiter_raw():
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- STT / TTS ---
@app.post("/stt")
async def stt(file: UploadFile):
    audio = await file.read()
    t = time.perf_counter()
    text = transcribe(audio)
    ms = (time.perf_counter() - t) * 1000
    print(f"STT {ms:.0f}ms  ({len(text)} chars)")
    return JSONResponse({"text": text}, headers={"X-STT-ms": f"{ms:.0f}"})


class TTSReq(BaseModel):
    text: str


@app.post("/tts")
def tts(req: TTSReq):
    t = time.perf_counter()
    wav = synth_wav(req.text)
    ms = (time.perf_counter() - t) * 1000
    audio_s = (len(wav) - 44) / 2 / 22050          # 16-bit mono @ 22.05kHz
    print(f"TTS {ms:.0f}ms  ({len(req.text)} chars -> {audio_s:.1f}s audio, {audio_s*1000/ms:.1f}x realtime)")
    return Response(wav, media_type="audio/wav",
                    headers={"X-TTS-ms": f"{ms:.0f}", "X-TTS-audio-s": f"{audio_s:.1f}"})


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
