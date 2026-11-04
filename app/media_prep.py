"""Local media helpers (ffmpeg/ffprobe + PyMuPDF).

Used for screenshot extraction, PDF page rendering and duration probes.
ffmpeg/ffprobe are required for video work; without them those features
raise a clear error instead of silently degrading.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def duration_sec(src: Path) -> float | None:
    """Media duration in seconds via ffprobe, or None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def pdf_pages(pdf: Path, workdir: Path, dpi: int = 100, max_pages: int = 100,
              log: LogFn | None = None) -> list[Path]:
    """Render PDF pages to JPEGs for image-only models."""
    try:
        import pymupdf
    except ImportError:
        raise RuntimeError("PDF pages need PyMuPDF: pip install pymupdf")
    doc = pymupdf.open(str(pdf))
    try:
        n = min(len(doc), max_pages)
        if len(doc) > max_pages and log:
            log(f"{pdf.name}: using first {max_pages} of {len(doc)} pages")
        workdir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for i in range(n):
            dest = workdir / f"{pdf.stem}_p{i + 1:03d}.jpg"
            doc[i].get_pixmap(dpi=dpi, alpha=False).save(str(dest))
            out.append(dest)
        if log:
            log(f"Rendered {n} pages from {pdf.name}")
        return out
    finally:
        doc.close()


def extract_frames(src: Path, workdir: Path, fps: float, max_frames: int = 200,
                   log: LogFn | None = None) -> list[Path]:
    """Extract screenshots from a video for image-only models.

    Frames are capped at max_frames (fps auto-lowers if needed) and
    near-duplicate consecutive frames are dropped (static slides be gone).
    Raises RuntimeError when ffmpeg is missing (explicit request: fail fast).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Screenshots need ffmpeg on PATH.")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    dur = duration_sec(src)
    if dur and dur * fps > max_frames:
        fps = max_frames / dur
        if log:
            log(f"Frame cap ({max_frames}): lowering screenshot fps to {fps:.3f}")
    workdir.mkdir(parents=True, exist_ok=True)
    pattern = str(workdir / f"{src.stem}_shot%04d.jpg")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-vf", f"fps={fps},scale=1280:-2", "-q:v", "4", pattern]
    if log:
        log(f"Extracting screenshots from {src.name} at {fps:.3f} fps…")
    subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    frames = sorted(workdir.glob(f"{src.stem}_shot*.jpg"))
    if log:
        log(f"Extracted {len(frames)} frames")
    kept = _dedupe_frames(frames, log)
    return kept


def _dedupe_frames(frames: list[Path], log: LogFn | None = None,
                   threshold: float = 3.0) -> list[Path]:
    """Drop consecutive near-identical frames (Pillow only, no numpy)."""
    if len(frames) < 2:
        return frames
    from PIL import Image, ImageChops, ImageStat
    kept = [frames[0]]
    prev = Image.open(frames[0]).convert("L").resize((32, 32))
    for f in frames[1:]:
        cur = Image.open(f).convert("L").resize((32, 32))
        diff = ImageStat.Stat(ImageChops.difference(prev, cur)).mean[0]
        if diff >= threshold:
            kept.append(f)
            prev = cur
        else:
            f.unlink(missing_ok=True)
    if log and len(kept) != len(frames):
        log(f"Deduped screenshots: {len(frames)} -> {len(kept)}")
    return kept
