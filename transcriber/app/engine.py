"""Local transcription engine (faster-whisper) with a cross-process lock.

Nothing leaves this machine: audio/video is decoded locally. CPU-friendly
(int8), ~99 languages, no API key. The first run downloads the model
(~1.6 GB for large-v3-turbo).

A GPU is used automatically when CTranslate2 can see one, otherwise it falls
back to CPU -- that covers both NVIDIA (CUDA) and AMD (ROCm) builds; the device
is called "cuda" either way. See the README section "AMD GPU (ROCm)".
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

DEFAULT_MODEL = config.MODEL

_model_cache: dict[str, object] = {}
_pipe_cache: dict[str, object] = {}

# CTranslate2's default allocator hits "Memory access fault" on longer audio on
# AMD/ROCm; the caching allocator is stable (verified up to 20-minute files).
# An explicit user value always wins; unused on CPU-only runs.
os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cub_caching")

_lock_init = threading.Lock()
_LOCK: object | None = None


@dataclass
class Result:
    text: str
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    segments: int = 0


def _gpu_lock():
    """One transcription at a time, across threads *and* processes.

    Two jobs sharing one GPU can exhaust VRAM, and a GPU memory fault is
    process-fatal, so requests are serialised. The lock file lives in DATA_DIR,
    so several server processes on the same machine (or a CLI) coordinate too.
    """
    global _LOCK
    with _lock_init:
        if _LOCK is None:
            from filelock import FileLock
            _LOCK = FileLock(str(config.DATA_DIR / "transcribe.lock"))
    return _LOCK


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def gpu_available() -> bool:
    """True when CTranslate2 exposes at least one GPU (CUDA or ROCm)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve(device: str | None = None, compute_type: str | None = None
            ) -> tuple[str, str]:
    """Resolve (device, compute_type), honouring args > env > auto.

    device "auto" -> GPU when present, else cpu. compute_type "auto" ->
    float16 on GPU, int8 on CPU.
    """
    dev = (device if device is not None else config.DEVICE) or "auto"
    dev = dev.strip().lower()
    if dev in ("", "auto"):
        dev = "cuda" if gpu_available() else "cpu"
    elif dev in ("gpu", "rocm"):
        dev = "cuda"
    elif dev not in ("cuda", "cpu"):
        dev = "cpu"
    ct = (compute_type if compute_type is not None else config.COMPUTE_TYPE) or "auto"
    ct = ct.strip().lower()
    if ct in ("", "auto"):
        ct = "float16" if dev == "cuda" else "int8"
    return dev, ct


def _load(model_name: str, device: str, compute_type: str, log: LogFn | None):
    key = f"{model_name}/{device}/{compute_type}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        if log:
            where = "GPU" if device == "cuda" else "CPU"
            log(f"Loading model {model_name} on {where} ({device}/{compute_type}; "
                f"first run downloads it, ~1.6 GB)…")
        _model_cache[key] = WhisperModel(model_name, device=device,
                                         compute_type=compute_type)
    return _model_cache[key]


def _batched(model, key: str):
    """BatchedInferencePipeline wrapper, cached per model key.

    Whisper decoding is sequential per segment, which starves a GPU. Batched
    mode chunks by VAD and decodes many segments in parallel -- on an RX 7600
    this measured ~75x realtime vs ~11x sequential. Trade-off: segments are
    decoded independently, so there is no cross-segment conditioning.
    """
    if key not in _pipe_cache:
        from faster_whisper import BatchedInferencePipeline
        _pipe_cache[key] = BatchedInferencePipeline(model=model)
    return _pipe_cache[key]


def _consume(segments, log: LogFn | None) -> tuple[str, int]:
    chunks: list[str] = []
    n = 0
    for seg in segments:
        chunks.append(seg.text.strip())
        n += 1
        if log and n % 50 == 0:
            log(f"… transcribed {n} segments")
    return " ".join(c for c in chunks if c), n


def transcribe_file(src: Path, model_name: str = DEFAULT_MODEL,
                    language: str | None = None,
                    device: str | None = None, compute_type: str | None = None,
                    batch_size: int | None = None,
                    log: LogFn | None = None) -> Result:
    """Transcribe one audio/video file locally.

    Serialised by a cross-process lock (see _gpu_lock), so concurrent requests
    queue instead of fighting over the GPU.
    """
    if not available():
        raise RuntimeError("Local transcription needs faster-whisper: "
                           "pip install -r requirements-transcribe.txt")
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    device, compute_type = resolve(device, compute_type)
    bs = int(batch_size if batch_size is not None else config.BATCH_SIZE)
    from filelock import Timeout
    lock = _gpu_lock()
    try:
        lock.acquire(timeout=0)
    except Timeout:
        if log:
            log("Another transcription is running — waiting for the GPU…")
        lock.acquire()
    try:
        return _transcribe(src, model_name, language, device, compute_type, bs, log)
    finally:
        lock.release()


def _transcribe(src: Path, model_name: str, language: str | None,
                device: str, compute_type: str, bs: int,
                log: LogFn | None) -> Result:
    """The actual decode (caller holds the transcription lock)."""
    model = _load(model_name, device, compute_type, log)
    key = f"{model_name}/{device}/{compute_type}"
    if log:
        lang = language or "auto-detect"
        where = "GPU" if device == "cuda" else "CPU"
        batch = f", batch={bs}" if bs > 0 else ""
        log(f"Transcribing {src.name} ({model_name}, {where} "
            f"{device}/{compute_type}{batch}, lang={lang})…")
    # vad_filter skips silence (lectures have pauses) and, in batched mode,
    # provides the chunks that get decoded in parallel. faster-whisper reads
    # audio streams straight from video containers, no extraction needed.
    common = dict(language=language, beam_size=5, vad_filter=True,
                  vad_parameters=dict(min_silence_duration_ms=500))
    if bs > 0:
        eng = _batched(model, key)
        try:
            segments, info = eng.transcribe(str(src), batch_size=bs, **common)
            text, n = _consume(segments, log)
        except Exception as e:
            if log:
                log(f"Batched transcription failed ({e}); retrying sequentially.")
            segments, info = model.transcribe(str(src), **common)
            text, n = _consume(segments, log)
    else:
        segments, info = model.transcribe(str(src), **common)
        text, n = _consume(segments, log)
    res = Result(
        text=text,
        language=str(getattr(info, "language", "") or ""),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=n)
    if log:
        log(f"Detected language: {res.language} (p={res.language_probability:.2f}), "
            f"audio length: {res.duration:.0f}s")
        log(f"Transcript: {len(text)} chars from {n} segments")
    return res
