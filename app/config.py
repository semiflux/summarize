"""Portable settings: env vars / .env only. No hardcoded paths."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

SERVER_API_KEY = os.environ.get("LLM_API_KEY", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
LLM_BASE = os.environ.get("LLM_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY = SERVER_API_KEY
LLM_MODEL = DEFAULT_MODEL
# Reasoning control for reasoning-capable models (OpenRouter `reasoning` param):
# level = "" (provider default) | minimal | low | medium | high | off.
REASONING = os.environ.get("LLM_REASONING", "")
# Hard ceiling on reasoning tokens (0 = no ceiling: use raw effort levels).
# OpenRouter accepts either `effort` or `max_tokens`, not both; we express levels
# as token budgets so every request is hard-bounded against runaway thinking.
REASONING_MAX_TOKENS = int(os.environ.get("LLM_REASONING_MAX_TOKENS", "65536"))
PORT = int(os.environ.get("PORT", "8000"))

# Upload limits (keep requests manageable for chat-completions endpoints)
MAX_FILES = 50
MAX_FILE_MB = 500
MAX_TOTAL_MB = 2000

TEXT_EXTS = {".txt", ".md", ".markdown", ".srt", ".vtt", ".csv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
# Audio/video are transcribed locally (faster-whisper) and/or screenshotted —
# this backend takes text+images only.
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Union of everything the pipeline accepts (server + CLI share this).
ALLOWED_EXTS = TEXT_EXTS | IMAGE_EXTS | PDF_EXTS | AUDIO_EXTS | VIDEO_EXTS

# Images larger than this are skipped for the chat backend (base64 blowup).
MAX_IMAGE_MB = 20

# How PDFs reach the model. Default: render each page to a JPEG and send it as
# an image — portable (any vision model, incl. local Ollama) and keeps slide
# figures. PDF_AS_FILE instead sends the whole PDF as one `file` content part
# and lets the provider parse it (OpenRouter/OpenAI). Cheaper for text-heavy
# documents, but OCR fallbacks can drop figures, and some providers bill per page.
PDF_AS_FILE = os.environ.get("PDF_AS_FILE", "").strip().lower() in {"1", "true", "yes", "on"}
# OpenRouter PDF parsing engine (only with PDF_AS_FILE; OpenRouter only):
# auto (native if the model supports files, else mistral-ocr) | native |
# mistral-ocr | cloudflare-ai.
PDF_ENGINE = os.environ.get("PDF_ENGINE", "auto").strip() or "auto"
# Providers cap whole-file inputs (~50 MB at OpenAI); larger PDFs fall back to
# page rendering.
PDF_MAX_MB = int(os.environ.get("PDF_MAX_MB", "50"))

# Default local transcription model (faster-whisper). Env-overridable.
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "large-v3-turbo")
# Transcription device: auto (GPU when CTranslate2 sees one, else CPU) | cpu | cuda.
# CTranslate2 calls the device "cuda" on both NVIDIA CUDA and AMD ROCm builds.
TRANSCRIBE_DEVICE = os.environ.get("TRANSCRIBE_DEVICE", "auto")
# Compute type: auto (float16 on GPU, int8 on CPU) | int8 | int8_float16 | float16 | float32.
TRANSCRIBE_COMPUTE_TYPE = os.environ.get("TRANSCRIBE_COMPUTE_TYPE", "auto")
# Decode segments in parallel (faster-whisper batched pipeline). Much faster on a
# GPU; 0 disables it and decodes sequentially.
TRANSCRIBE_BATCH_SIZE = int(os.environ.get("TRANSCRIBE_BATCH_SIZE", "16"))

# Screenshot extraction cap per video (fps auto-lowers to respect it).
SHOT_MAX = int(os.environ.get("SHOT_MAX", "200"))
