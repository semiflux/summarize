"""Portable CLI (no UI): python cli.py <files...> [--out DIR] [--api-key KEY]"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from app import config
from app.pipeline import new_job_dir, run_job, job_info, _safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize lesson files to LaTeX PDF (OpenAI-compatible LLM).")
    ap.add_argument("input_files", type=Path, nargs="+",
                    help="Input files or directories (directories expand recursively "
                         "to supported files, sorted, hidden paths skipped).")
    ap.add_argument("--out", type=Path, default=Path("./data/jobs-cli"),
                    help="Destination folder for the finished .tex/.pdf files.")
    ap.add_argument("--api-key", default="", help="LLM API key (or set LLM_API_KEY).")
    ap.add_argument("--model", default=None,
                    help="Model id (default: nvidia/nemotron-3-ultra-550b-a55b:free).")
    ap.add_argument("--base-url", default=None,
                    help="API base URL (default: https://openrouter.ai/api/v1).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--language", default="auto", choices=["auto", "it", "en"])
    ap.add_argument("--primary", default="",
                    help="Comma-separated 1-based positions and/or filename substrings "
                         "marking the PRIMARY lesson sources, e.g. --primary 1,transcript. "
                         "Omit for auto-detect.")
    ap.add_argument("--transcribe-model", default=None,
                    help="Local transcription model for audio/video (default: large-v3-turbo; "
                         "needs requirements-transcribe.txt).")
    ap.add_argument("--transcribe-lang", default="auto",
                    help="Transcript language: auto, it, en, … (default: auto).")
    ap.add_argument("--transcribe-device", default=None,
                    choices=["auto", "cpu", "cuda"],
                    help="Transcription device (default: auto — GPU when "
                         "CTranslate2 sees one, else CPU; 'cuda' also means AMD "
                         "ROCm). Use 'cpu' to force CPU.")
    ap.add_argument("--transcribe-compute-type", default=None,
                    choices=["auto", "int8", "int8_float16", "float16", "float32"],
                    help="Transcription compute type (default: auto — float16 on "
                         "GPU, int8 on CPU).")
    ap.add_argument("--transcribe-batch-size", dest="transcribe_batch_size",
                    type=int, default=None,
                    help="Decode this many VAD segments in parallel (default: "
                         "TRANSCRIBE_BATCH_SIZE, 16). Much faster on a GPU; 0 turns "
                         "the batched pipeline off.")
    ap.add_argument("--screenshot", "--screenshots", dest="screenshot",
                    type=float, default=0.0,
                    help="Extract video frames as photos at this fps (0 = off). Frames are "
                         "added on top of the transcript; near-duplicates are auto-dropped. "
                         "Needs ffmpeg.")
    ap.add_argument("--no-transcript", dest="transcribe", action="store_false",
                    help="Skip local transcription of audio/video. Requires --screenshot "
                         "(otherwise there is nothing to send).")
    ap.add_argument("--reasoning", default=None,
                    choices=["low", "medium", "high", "off"],
                    help="Reasoning level for reasoning-capable models (OpenRouter "
                         "`reasoning` param). Omit for the provider default. Reasoning "
                         "traces are saved as reasoning.md; reasoning tokens are billed "
                         "as output.")
    ap.add_argument("--reasoning-max-tokens", dest="reasoning_max_tokens",
                    type=int, default=None,
                    help="Hard ceiling on reasoning tokens (default: 65536, "
                         "LLM_REASONING_MAX_TOKENS). The level maps to a token budget "
                         "clamped to this ceiling, which prevents runaway thinking "
                         "(the cause of empty outputs). 0 = no ceiling: send raw "
                         "effort levels instead.")
    ap.add_argument("--as-file", dest="pdf_as_file", action="store_true", default=None,
                    help="Send PDFs whole as a provider-parsed `file` part instead of "
                         "rendering pages to images (OpenRouter/OpenAI). Cheaper for "
                         "text-heavy documents, but OCR fallbacks can drop slide "
                         "figures — see --pdf-engine. Default: PDF_AS_FILE env.")
    ap.add_argument("--pdf-engine", dest="pdf_engine", default=None,
                    choices=["auto", "native", "mistral-ocr", "cloudflare-ai"],
                    help="OpenRouter PDF parsing engine for --as-file: auto (native "
                         "if the model supports files, else mistral-ocr; default), "
                         "native, mistral-ocr (billed per page), cloudflare-ai (free).")
    args = ap.parse_args()

    if not args.transcribe and not args.screenshot:
        ap.error("--no-transcript needs --screenshot FPS: a raw audio/video file can't be "
                 "sent to a text+images API, so nothing would remain.")

    valid: list[Path] = []
    for p in args.input_files:
        if p.is_dir():
            valid.extend(sorted(
                q for q in p.rglob("*")
                if q.is_file()
                and q.suffix.lower() in config.ALLOWED_EXTS
                and not any(part.startswith(".") for part in q.relative_to(p).parts)))
        elif p.is_file():
            valid.append(p)
    if not valid:
        raise SystemExit("No valid input files.")
    job_dir = new_job_dir()
    want: set[int] = set()
    for tok in (args.primary or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            want.add(int(tok) - 1)
        else:
            for i, p in enumerate(valid):
                if tok.lower() in p.name.lower():
                    want.add(i)
    primary_names: list[str] = []
    for i, p in enumerate(valid):
        safe = _safe_filename(p.name)
        dest = job_dir / "uploads" / f"{i:02d}_{safe}"
        n = 1
        while dest.exists():
            dest = job_dir / "uploads" / f"{i:02d}_{dest.stem}_{n}{dest.suffix}"
            n += 1
        shutil.copy(p, dest)
        if i in want:
            primary_names.append(dest.name)
    run_job(job_dir, model=args.model, api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature, language=args.language,
            primary_names=primary_names,
            transcribe_model=args.transcribe_model or config.TRANSCRIBE_MODEL,
            transcribe_lang=args.transcribe_lang,
            transcribe_device=args.transcribe_device,
            transcribe_compute_type=args.transcribe_compute_type,
            transcribe_batch_size=args.transcribe_batch_size,
            screenshots_fps=args.screenshot or None,
            transcribe=args.transcribe,
            reasoning=args.reasoning or config.REASONING,
            reasoning_max_tokens=args.reasoning_max_tokens,
            pdf_as_file=(args.pdf_as_file if args.pdf_as_file is not None
                         else config.PDF_AS_FILE),
            pdf_engine=args.pdf_engine or config.PDF_ENGINE)
    info = job_info(job_dir)
    print(f"status: {info['status']}")
    u, c = info.get("usage", {}), info.get("cost", {})
    print(f"tokens: in {u.get('prompt')} out {u.get('output')} "
          f"thoughts={u.get('thoughts')} "
          f"modalities={u.get('by_modality')} calls={u.get('calls')}")
    print(f"cost estimate: ${c.get('total_usd')} ({c.get('note')})")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for a in info["artifacts"]:
        src = job_dir / a["name"]
        if not src.is_file():
            src = job_dir / "transcripts" / a["name"]
        shutil.copy(src, out / a["name"])
        print(f"saved: {(out / a['name']).resolve()}")
    print(job_dir.joinpath("job.log").read_text()[-3000:])


if __name__ == "__main__":
    main()
