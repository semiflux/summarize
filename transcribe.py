"""Standalone local transcription (no Gemini, no API key).

Usage:
    python transcribe.py lecture.mp3 lezione.webm --out ./transcripts
    python transcribe.py ./audio-folder --model large-v3 --lang it

Needs: pip install -r requirements-transcribe.txt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app import config, transcribe_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDIA_EXTS = config.AUDIO_EXTS | config.VIDEO_EXTS


def collect(inputs: list[Path]) -> list[Path]:
    valid: list[Path] = []
    for p in inputs:
        if p.is_dir():
            valid.extend(sorted(
                q for q in p.rglob("*")
                if q.is_file()
                and q.suffix.lower() in MEDIA_EXTS
                and not any(part.startswith(".") for part in q.relative_to(p).parts)))
        elif p.is_file():
            if p.suffix.lower() not in MEDIA_EXTS:
                logger.warning("Skipping unsupported file: %s", p.name)
                continue
            valid.append(p)
    return valid


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe audio/video locally with faster-whisper.")
    ap.add_argument("input_files", type=Path, nargs="+",
                    help="Audio/video files or directories (recursive, sorted).")
    ap.add_argument("--out", type=Path, default=Path("./transcripts"),
                    help="Destination folder for .txt transcripts.")
    ap.add_argument("--model", default=config.TRANSCRIBE_MODEL,
                    help="faster-whisper model (default: large-v3-turbo).")
    ap.add_argument("--lang", default="auto",
                    help="Transcript language: auto, it, en, … (default: auto).")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="Inference device (default: TRANSCRIBE_DEVICE or auto — "
                         "GPU when CTranslate2 sees one, else CPU). 'cuda' is the "
                         "name on both NVIDIA CUDA and AMD ROCm builds.")
    ap.add_argument("--compute-type", default=None,
                    choices=["auto", "int8", "int8_float16", "float16", "float32"],
                    help="Compute type (default: auto — float16 on GPU, int8 on CPU).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Decode this many VAD segments in parallel (default: "
                         "TRANSCRIBE_BATCH_SIZE, 16). Much faster on a GPU; "
                         "0 = sequential.")
    args = ap.parse_args()

    valid = collect(args.input_files)
    if not valid:
        raise SystemExit("No audio/video files found.")
    args.out.mkdir(parents=True, exist_ok=True)
    lang = None if args.lang == "auto" else args.lang
    dev, ct = transcribe_local.resolve(args.device, args.compute_type)
    bs = config.TRANSCRIBE_BATCH_SIZE if args.batch_size is None else args.batch_size
    print(f"device: {dev}/{ct}" + (f", batch={bs}" if bs > 0 else ""))

    for p in valid:
        dest = args.out / f"{p.stem}.txt"
        n = 1
        while dest.exists():
            dest = args.out / f"{p.stem}_{n}.txt"
            n += 1
        text = transcribe_local.transcribe_file(p, model_name=args.model,
                                                language=lang,
                                                device=args.device,
                                                compute_type=args.compute_type,
                                                batch_size=args.batch_size,
                                                log=logger.info)
        dest.write_text(text, encoding="utf-8")
        print(f"saved: {dest.resolve()} ({len(text)} chars)")


if __name__ == "__main__":
    main()
