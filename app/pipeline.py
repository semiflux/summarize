"""Job pipeline: save uploads -> LLM (text+images) -> sanitise -> compile PDF.

Audio/video are transcribed locally (faster-whisper); video is optionally
screenshotted; PDFs are rendered to page images. The LLM only ever sees
text + images through an OpenAI-compatible chat API.
"""
from __future__ import annotations

import logging
import re
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config, llm_openai
from .prompts import SYSTEM_PROMPT
from .latex_utils import sanitise_latex, compile_pdf, create_dark_pdf, LatexProcessingError, title_slug, find_tex

logger = logging.getLogger(__name__)


def _safe_filename(original: str) -> str:
    """ASCII-safe filename for maximum portability across systems/APIs."""
    name = Path(original or "file").name
    stem, ext = (name.rsplit(".", 1) + [""])[:2]
    ext = "." + re.sub(r"[^A-Za-z0-9]", "", ext or "") if "." in name else ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "file"
    return stem[:100] + ext.lower()[:10]


def _add_usage(job_dir: Path, usage: dict, call: str) -> dict:
    """Accumulate token usage across generate + fix calls into usage.json."""
    import json
    path = job_dir / "usage.json"
    total: dict = {"prompt": 0, "output": 0, "thoughts": 0, "total": 0,
                   "by_modality": {}, "calls": 0}
    if path.exists():
        try:
            total.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    for k in ("prompt", "output", "thoughts", "total"):
        total[k] = int(total.get(k, 0)) + int(usage.get(k, 0))
    mods = total.get("by_modality", {})
    for mod, n in (usage.get("by_modality") or {}).items():
        mods[mod] = int(mods.get(mod, 0)) + int(n)
    total["by_modality"] = mods
    total["calls"] = int(total.get("calls", 0)) + 1
    total["last_call"] = call
    if usage.get("finish_reason"):
        total["finish_reason"] = usage["finish_reason"]
    path.write_text(json.dumps(total), encoding="utf-8")
    return total


def _save_reasoning(job_dir: Path, trace: list[str], log) -> None:
    """Persist the model's reasoning trace (when a reasoning level is set)."""
    text = "\n\n---\n\n".join(t for t in trace if t and t.strip())
    if not text:
        return
    (job_dir / "reasoning.md").write_text(text, encoding="utf-8")
    log(f"Saved model reasoning to reasoning.md ({len(text)} chars)")


def new_job_dir() -> Path:
    jid = uuid.uuid4().hex[:12]
    d = config.DATA_DIR / "jobs" / jid
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    return d


def run_job(job_dir: Path, model: str | None = None, api_key: str = "",
            base_url: str | None = None, temperature: float = 0.2,
            language: str = "auto", primary_names: list[str] | None = None,
            transcribe_model: str | None = None, transcribe_lang: str | None = None,
            transcribe_device: str | None = None,
            transcribe_compute_type: str | None = None,
            transcribe_batch_size: int | None = None,
            screenshots_fps: float | None = None,
            transcribe_only: bool = False, reasoning: str = "",
            reasoning_max_tokens: int | None = None,
            transcribe: bool = True, pdf_as_file: bool = False,
            pdf_engine: str = "") -> None:
    log_path = job_dir / "job.log"
    status_path = job_dir / "status.txt"

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def set_status(s: str) -> None:
        status_path.write_text(s, encoding="utf-8")

    try:
        set_status("running")
        uploads = sorted((job_dir / "uploads").iterdir())
        files = [p for p in uploads if p.is_file()]
        if not files:
            raise ValueError("No files uploaded.")
        log(f"Files: {[p.name for p in files]}")
        if primary_names:
            log(f"User-designated primary sources: {primary_names}")
        if transcribe_only:
            transcribe = True
            screenshots_fps = None  # only transcripts wanted
        if not screenshots_fps and not transcribe:
            raise ValueError("Nothing to send: --no-transcript needs --screenshot FPS "
                             "(a raw audio/video file can't be sent to a text+images API).")
        rmax = config.REASONING_MAX_TOKENS if reasoning_max_tokens is None else int(reasoning_max_tokens)
        (job_dir / "params.json").write_text(
            __import__("json").dumps({"model": model, "temperature": temperature,
                                       "language": language,
                                       "primary_names": primary_names or [],
                                       "transcribe_model": transcribe_model,
                                       "transcribe_lang": transcribe_lang,
                                       "transcribe_device": transcribe_device,
                                       "transcribe_compute_type": transcribe_compute_type,
                                       "transcribe_batch_size": transcribe_batch_size,
                                       "screenshots_fps": screenshots_fps,
                                       "transcribe_only": transcribe_only,
                                       "transcribe": transcribe,
                                       "reasoning": reasoning,
                                       "reasoning_max_tokens": rmax,
                                       "pdf_as_file": pdf_as_file,
                                       "pdf_engine": pdf_engine,
                                       "base_url": base_url}),
            encoding="utf-8")
        # Video handling: frames-as-photos and audio transcription are independent
        # and can be combined. Default is transcript-only; --screenshot adds frames.
        ordered: list[Path] = []
        if screenshots_fps:
            from . import media_prep
            sdir = job_dir / "shots"
            for p in files:
                ordered.append(p)
                if p.suffix.lower() not in config.VIDEO_EXTS:
                    continue
                try:
                    frames = media_prep.extract_frames(
                        p, sdir, screenshots_fps, config.SHOT_MAX, log)
                except Exception as e:
                    log(f"Screenshot extraction failed for {p.name}: {e}")
                    raise
                if frames:
                    ordered.extend(frames)
                else:
                    log(f"WARNING: no frames from {p.name}")
        else:
            ordered = list(files)
        files = ordered

        # Local transcription: every audio/video, independent of screenshots.
        exclude: set[str] = set()
        extra_texts: list[tuple[str, str]] = []
        media = [p for p in files
                 if p.suffix.lower() in (config.AUDIO_EXTS | config.VIDEO_EXTS)]
        if media and transcribe:
            from . import transcribe_local
            tdir = job_dir / "transcripts"
            tdir.mkdir(exist_ok=True)
            lang = transcribe_lang if transcribe_lang and transcribe_lang != "auto" else None
            for p in media:
                try:
                    text = transcribe_local.transcribe_file(
                        p, model_name=transcribe_model or transcribe_local.DEFAULT_MODEL,
                        language=lang, device=transcribe_device,
                        compute_type=transcribe_compute_type,
                        batch_size=transcribe_batch_size, log=log)
                except Exception as e:
                    log(f"Local transcription failed for {p.name}: {e}")
                    raise
                if text.strip():
                    (tdir / f"{p.stem}.txt").write_text(text, encoding="utf-8")
                    extra_texts.append((p.name, text))
                    exclude.add(p.name)
                else:
                    log(f"WARNING: empty transcript for {p.name} — excluding it "
                        f"(this backend cannot take raw audio/video)")
                    exclude.add(p.name)
        elif media:
            # --no-transcript: raw media can't be sent, so it must be represented
            # by its frames (--screenshot). Drop it from the content otherwise.
            for p in media:
                log(f"Not transcribing {p.name} (--no-transcript); using its frames.")
                exclude.add(p.name)

        if transcribe_only:
            if not extra_texts:
                raise ValueError("Nothing to transcribe (no audio/video with usable speech).")
            log(f"Transcribe-only mode: {len(extra_texts)} transcript(s), skipping notes.")
            set_status("done")
            return

        texts, images, documents = llm_openai.collect_texts_images(
            files, frozenset(exclude), extra_texts, job_dir, log, as_file=pdf_as_file)
        labels = ([n for n, _ in texts] + [lb for lb, _, _ in images]
                  + [lb for lb, _, _ in documents])
        lang_line = "" if language == "auto" else f"\n- output language: {language}."
        override_line = ""
        if primary_names:
            matched = [lb for lb in labels for pn in primary_names
                       if lb == pn or lb.startswith(pn + " ")
                       or lb.startswith(pn.rsplit(".", 1)[0] + "_shot")]
            shown = matched or list(primary_names)
            override_line = ("\n- source override: the user explicitly designated these files "
                             "as the PRIMARY lesson sources: "
                             + ", ".join(shown)
                             + ". respect this designation instead of inferring primacy yourself; "
                               "treat everything else as integrative.")
        system = SYSTEM_PROMPT + lang_line + override_line
        trace: list[str] = []
        raw, usage = llm_openai.generate_notes(
            system, texts, images,
            model=model or config.DEFAULT_MODEL,
            api_key=api_key, base_url=base_url or config.LLM_BASE,
            temperature=temperature, log=log,
            reasoning=reasoning or config.REASONING,
            reasoning_max_tokens=rmax, trace=trace,
            documents=documents, pdf_engine=pdf_engine or config.PDF_ENGINE)
        _save_reasoning(job_dir, trace, log)
        total_usage = _add_usage(job_dir, usage, "generate")
        log(f"Tokens: prompt={usage.get('prompt')} output={usage.get('output')}")
        if usage.get("thoughts"):
            log(f"Reasoning tokens: {usage['thoughts']}")
        if not raw.strip():
            fr = usage.get("finish_reason")
            rt, ot = int(usage.get("thoughts", 0)), int(usage.get("output", 0))
            hint = ""
            if fr:
                hint += f" finish_reason={fr}."
            if rt and ot and rt >= ot:
                hint += (f" The entire output budget was spent reasoning "
                         f"({rt}/{ot} tokens) — lower the reasoning level or the "
                         f"reasoning token cap (LLM_REASONING_MAX_TOKENS), or add "
                         f"transcripts so the model has less to infer from images.")
            raise ValueError("Model returned empty output." + hint)
        lang = None if language == "auto" else language
        tex = sanitise_latex(raw, preferred_language=lang)
        slug = title_slug(tex, fallback=files[0].stem if files else "lecture_notes")
        tex_path = job_dir / f"{slug}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        log(f"Wrote {tex_path.name} ({len(tex)} chars)")

        compiled = None
        try:
            compiled = compile_pdf(tex_path)
            log(f"PDF OK: {compiled.name}")
        except LatexProcessingError as e:
            (job_dir / "compile_errors.log").write_text(str(e)[-8000:], encoding="utf-8")
            log(f"Compile failed: {str(e)[-500:]}")
        if compiled is not None:
            dark = create_dark_pdf(compiled)
            if dark:
                log(f"Dark PDF OK: {dark.name}")
            set_status("done")
        else:
            log("PDF compilation failed. .tex kept — use 'Retry LaTeX fix' to re-run manually.")
            set_status("done_no_pdf")
    except Exception as e:
        traceback.print_exc()
        log(f"ERROR: {e}")
        set_status(f"error: {e}")


def retry_fix(job_dir: Path, model: str | None = None,
              temperature: float | None = None, api_key: str = "",
              base_url: str | None = None, reasoning: str = "",
              reasoning_max_tokens: int | None = None) -> None:
    """Manual fix pass: send current .tex + saved compiler errors back to the model, once."""
    import json

    log_path = job_dir / "job.log"
    status_path = job_dir / "status.txt"

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    params: dict = {}
    if (job_dir / "params.json").exists():
        params = json.loads((job_dir / "params.json").read_text(encoding="utf-8"))
    model = model or params.get("model") or config.DEFAULT_MODEL
    temperature = params.get("temperature", 0.2) if temperature is None else temperature
    base_url = base_url or params.get("base_url") or config.LLM_BASE
    reasoning = reasoning or params.get("reasoning") or config.REASONING
    if reasoning_max_tokens is None:
        reasoning_max_tokens = params.get("reasoning_max_tokens")
    rmax = config.REASONING_MAX_TOKENS if reasoning_max_tokens is None else int(reasoning_max_tokens)
    language = params.get("language", "auto")
    lang = None if language == "auto" else language

    tex_path = find_tex(job_dir)
    if tex_path is None:
        raise ValueError("No .tex to fix yet.")
    errors = (job_dir / "compile_errors.log").read_text(encoding="utf-8") if (job_dir / "compile_errors.log").exists() else ""
    status_path.write_text("fixing", encoding="utf-8")
    try:
        log(f"Manual fix pass with {model}…")
        trace: list[str] = []
        fixed_raw, usage = llm_openai.fix_latex(
            tex_path.read_text(encoding="utf-8"), errors,
            model=model, api_key=api_key, base_url=base_url,
            temperature=float(temperature), log=log,
            reasoning=reasoning, reasoning_max_tokens=rmax, trace=trace)
        _save_reasoning(job_dir, trace, log)
        _add_usage(job_dir, usage, "fix")
        if not fixed_raw.strip():
            fr = usage.get("finish_reason")
            rt, ot = int(usage.get("thoughts", 0)), int(usage.get("output", 0))
            hint = f" finish_reason={fr}." if fr else ""
            if rt and ot and rt >= ot:
                hint += (f" All {ot} output tokens were reasoning — lower the "
                         f"reasoning level or the reasoning token cap.")
            raise ValueError("Fix pass returned empty output." + hint)
        tex = sanitise_latex(fixed_raw, preferred_language=lang)
        tex_path.write_text(tex, encoding="utf-8")
        log(f"Fix wrote new .tex ({len(tex)} chars), recompiling…")
        try:
            pdf = compile_pdf(tex_path)
            log(f"PDF OK after manual fix: {pdf.name}")
            dark = create_dark_pdf(pdf)
            if dark:
                log(f"Dark PDF OK: {dark.name}")
            status_path.write_text("done", encoding="utf-8")
        except LatexProcessingError as e:
            (job_dir / "compile_errors.log").write_text(str(e)[-8000:], encoding="utf-8")
            log(f"Recompile still failing: {str(e)[-500:]}")
            status_path.write_text("done_no_pdf", encoding="utf-8")
    except Exception as e:
        traceback.print_exc()
        log(f"ERROR in fix pass: {e}")
        status_path.write_text("done_no_pdf", encoding="utf-8")
        raise


def _params_model(job_dir: Path) -> str:
    import json
    if (job_dir / "params.json").exists():
        try:
            p = json.loads((job_dir / "params.json").read_text(encoding="utf-8"))
            return p.get("model") or config.DEFAULT_MODEL
        except Exception:
            pass
    return config.DEFAULT_MODEL


def job_info(job_dir: Path) -> dict:
    status = (job_dir / "status.txt").read_text(encoding="utf-8") if (job_dir / "status.txt").exists() else "unknown"
    log = (job_dir / "job.log").read_text(encoding="utf-8", errors="replace") if (job_dir / "job.log").exists() else ""
    compile_errors = (job_dir / "compile_errors.log").read_text(encoding="utf-8", errors="replace") if (job_dir / "compile_errors.log").exists() else ""
    usage: dict = {"prompt": 0, "output": 0, "thoughts": 0, "total": 0, "by_modality": {}, "calls": 0}
    if (job_dir / "usage.json").exists():
        try:
            usage.update(__import__("json").loads((job_dir / "usage.json").read_text(encoding="utf-8")))
        except Exception:
            pass
    cost = llm_openai.estimate_cost(usage, _params_model(job_dir))
    artifacts = []
    for p in sorted(job_dir.glob("*.tex")) + sorted(job_dir.glob("*.pdf")):
        artifacts.append({"name": p.name, "size": p.stat().st_size,
                          "url": f"/api/jobs/{job_dir.name}/files/{p.name}"})
    tdir = job_dir / "transcripts"
    if tdir.exists():
        for p in sorted(tdir.glob("*.txt")):
            artifacts.append({"name": p.name, "size": p.stat().st_size,
                              "url": f"/api/jobs/{job_dir.name}/files/{p.name}"})
    rfile = job_dir / "reasoning.md"
    if rfile.exists():
        artifacts.append({"name": rfile.name, "size": rfile.stat().st_size,
                          "url": f"/api/jobs/{job_dir.name}/files/{rfile.name}"})
    uploads = [p.name for p in (job_dir / "uploads").iterdir()] if (job_dir / "uploads").exists() else []
    return {"id": job_dir.name, "status": status, "log": log[-8000:],
            "compile_errors": compile_errors[-6000:],
            "usage": usage, "cost": cost,
            "artifacts": artifacts, "uploads": uploads}
