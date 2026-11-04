"""Optional local transcription via faster-whisper (large-v3-turbo default).

Kept out of requirements.txt on purpose: `pip install -r requirements-transcribe.txt`
enables it. First run downloads the model (~1.6 GB for large-v3-turbo).
CPU-friendly (int8), 99 languages, no API key, no uploads, private.

A GPU is used automatically when CTranslate2 can see one, otherwise it falls
back to CPU. That covers both NVIDIA (CUDA) and AMD (ROCm) builds of
CTranslate2 -- the device is called "cuda" either way. See the README section
"AMD GPU (ROCm)" for the one-time ROCm wheel install.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from . import config

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

DEFAULT_MODEL = "large-v3-turbo"

_model_cache: dict[str, object] = {}
_pipe_cache: dict[str, object] = {}

# CTranslate2's default allocator hits "Memory access fault" on longer audio on
# AMD/ROCm; the caching allocator is stable (verified up to 20-minute files).
# An explicit user value always wins; unused on CPU-only runs.
os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cub_caching")


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
    dev = (device if device is not None else config.TRANSCRIBE_DEVICE) or "auto"
    dev = dev.strip().lower()
    if dev in ("", "auto"):
        dev = "cuda" if gpu_available() else "cpu"
    elif dev in ("gpu", "rocm"):
        dev = "cuda"
    elif dev not in ("cuda", "cpu"):
        dev = "cpu"
    ct = (compute_type if compute_type is not None
          else config.TRANSCRIBE_COMPUTE_TYPE) or "auto"
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
            log(f"Loading local transcription model {model_name} on {where} "
                f"({device}/{compute_type}; first run downloads it, ~1.6 GB)…")
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
                    log: LogFn | None = None) -> str:
    """Transcribe an audio/video file locally. Returns plain transcript text.

    device/compute_type default to auto (see resolve()); pass "cpu"/"int8" to
    force the CPU path. batch_size > 0 (default TRANSCRIBE_BATCH_SIZE) uses
    faster-whisper's batched pipeline; 0 disables it.
    """
    if not available():
        raise RuntimeError(
            "Local transcription needs faster-whisper: "
            "pip install -r requirements-transcribe.txt")
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    device, compute_type = resolve(device, compute_type)
    bs = int(batch_size if batch_size is not None
             else config.TRANSCRIBE_BATCH_SIZE)
    model = _load(model_name, device, compute_type, log)
    key = f"{model_name}/{device}/{compute_type}"
    if log:
        lang = language or "auto-detect"
        where = "GPU" if device == "cuda" else "CPU"
        batch = f", batch={bs}" if bs > 0 else ""
        log(f"Transcribing {src.name} locally ({model_name}, {where} "
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
    if log:
        detected = getattr(info, "language", "?")
        prob = getattr(info, "language_probability", 0.0)
        dur = getattr(info, "duration", 0.0)
        log(f"Detected language: {detected} (p={prob:.2f}), audio length: {dur:.0f}s")
        log(f"Local transcript: {len(text)} chars from {n} segments")
    return text
