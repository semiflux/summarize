"""Settings from env vars / .env. No hardcoded paths, no API keys."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load this service's own .env only (not the summarizer's one directory up).
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Speech model. large-v3-turbo is the best speed/quality trade-off for lectures.
MODEL = os.environ.get("TRANSCRIBE_MODEL", "large-v3-turbo")
# Device: auto (GPU when CTranslate2 sees one, else CPU) | cpu | cuda.
# CTranslate2 calls the device "cuda" on both NVIDIA CUDA and AMD ROCm builds.
DEVICE = os.environ.get("TRANSCRIBE_DEVICE", "auto")
# Compute type: auto (float16 on GPU, int8 on CPU) | int8 | int8_float16 |
# float16 | float32.
COMPUTE_TYPE = os.environ.get("TRANSCRIBE_COMPUTE_TYPE", "auto")
# Decode N VAD segments in parallel (faster-whisper batched pipeline); 0 =
# sequential. Much faster on a GPU.
BATCH_SIZE = int(os.environ.get("TRANSCRIBE_BATCH_SIZE", "16"))

# Bind address for `python -m app.main` (uvicorn's own --host/--port win
# otherwise). 127.0.0.1 is local-only; bind another address to share.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8100"))

# Per-request upload cap (MB).
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "2000"))

# Delete finished jobs (files + registry entry) after this many hours; 0 keeps
# them forever.
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS
