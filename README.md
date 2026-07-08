# VoiceUI

Local voice chat with LLMs. Browser UI → FastAPI → **llama.cpp** (chat),
**faster-whisper** (STT), **Piper** (TTS). All local, CPU-fine, no cloud.

## Setup

```bash
uv sync                                          # backend deps
./fetch-assets.sh                                # vendor Silero-VAD + onnxruntime-web into static/
python -m piper.download_voices en_US-lessac-medium --download-dir voices  # TTS voice
```

Get `llama-server` (prebuilt from ggml-org/llama.cpp releases, or Docker), then run it
with any GGUF model:

```bash
llama-server -m models/your-model.gguf --port 8080
```

## Run

```bash
uv run python main.py           # serves http://localhost:8000
```

Open <http://localhost:8000>. Type, or hold the 🎤 button (or spacebar) to talk;
switch **Voice → Hands-free** for always-listening VAD. "Speak replies" plays TTS.

## Config (env vars)

| var | default | note |
|-----|---------|------|
| `LLAMA_URL` | `http://localhost:8080/v1` | llama-server OpenAI endpoint |
| `WHISPER_MODEL` | `small` | `base`/`tiny` for lower latency |
| `WHISPER_COMPUTE` | `int8` | CTranslate2 compute type |
| `PIPER_VOICE` | `voices/en_US-lessac-medium.onnx` | TTS voice model |
| `SYSTEM_PROMPT` | *(none)* | prepended system message |

## Verify

```bash
uv run python main.py --selftest   # TTS 'hello world' -> STT round-trip
```
