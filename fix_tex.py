"""Standalone LaTeX fixer: compile -> send errors to the model -> recompile.

Usage:
    python fix_tex.py notes.tex --rounds 3 --api-key $LLM_API_KEY
    python fix_tex.py notes.tex --model qwen/qwen3.8-flash --api-key $LLM_API_KEY

Writes <stem>.fixed.tex (and .pdf on success) next to the original.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from app import config
from app.latex_utils import find_tex
from app.pipeline import retry_fix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix a .tex file until it compiles (or rounds run out).")
    ap.add_argument("tex_file", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--api-key", default="", help="LLM API key (or LLM_API_KEY).")
    ap.add_argument("--model", default=None, help="Model id.")
    ap.add_argument("--base-url", default=None, help="API base URL.")
    ap.add_argument("--reasoning", default=None,
                    choices=["low", "medium", "high", "off"],
                    help="Reasoning level for reasoning-capable models.")
    ap.add_argument("--reasoning-max-tokens", dest="reasoning_max_tokens",
                    type=int, default=None,
                    help="Hard ceiling on reasoning tokens (default: "
                         "LLM_REASONING_MAX_TOKENS, 65536). 0 = no ceiling.")
    args = ap.parse_args()

    src = args.tex_file
    if not src.is_file() or src.suffix.lower() != ".tex":
        raise SystemExit("Give me a .tex file.")

    with tempfile.TemporaryDirectory(prefix="fixjob-") as tmp:
        job = Path(tmp)
        shutil.copy(src, job / src.name)
        (job / "params.json").write_text(json.dumps({
            "model": args.model or config.DEFAULT_MODEL,
            "temperature": 0.2, "language": "auto",
            "base_url": args.base_url or config.LLM_BASE,
            "reasoning": args.reasoning or config.REASONING,
            "reasoning_max_tokens": config.REASONING_MAX_TOKENS
            if args.reasoning_max_tokens is None else args.reasoning_max_tokens,
        }), encoding="utf-8")
        for i in range(1, args.rounds + 1):
            print(f"--- fix round {i}/{args.rounds} ---")
            try:
                retry_fix(job, api_key=args.api_key)
            except Exception as e:
                print(f"fix pass failed: {e}")
                break
            if list(job.glob("*.pdf")):
                break
        tex = find_tex(job)
        if tex is None:
            raise SystemExit("No .tex produced.")
        out_tex = src.with_name(f"{src.stem}.fixed.tex")
        shutil.copy(tex, out_tex)
        print(f"saved: {out_tex.resolve()}")
        for pdf in sorted(job.glob("*.pdf")):
            tag = "" if pdf.stem == tex.stem else "_dark" if pdf.stem.endswith("_dark") else f"_{pdf.stem}"
            out_pdf = src.with_name(f"{src.stem}.fixed{tag}.pdf")
            shutil.copy(pdf, out_pdf)
            print(f"saved: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
