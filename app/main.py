"""FastAPI server: serves web UI + job API. Portable, no build step."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, pipeline
from .pipeline import _safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Lesson Summarizer")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
def get_config():
    tx = None
    try:
        from . import transcribe_local
        if transcribe_local.available():
            dev, ct = transcribe_local.resolve()
            bs = config.TRANSCRIBE_BATCH_SIZE
            tx = {"device": dev, "compute_type": ct,
                  "batch_size": bs,
                  "gpu": transcribe_local.gpu_available()}
    except Exception:
        pass
    return {"has_server_key": bool(config.SERVER_API_KEY),
            "default_model": config.DEFAULT_MODEL,
            "default_base_url": config.LLM_BASE,
            "default_reasoning": config.REASONING,
            "default_reasoning_max_tokens": config.REASONING_MAX_TOKENS,
            "default_pdf_as_file": config.PDF_AS_FILE,
            "default_pdf_engine": config.PDF_ENGINE,
            "transcribe": tx}


ALLOWED_EXTS = config.ALLOWED_EXTS


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     files: list[UploadFile] = File(...),
                     api_key: Optional[str] = Form(None),
                     model: Optional[str] = Form(None),
                     base_url: Optional[str] = Form(None),
                     temperature: float = Form(0.2),
                     language: str = Form("auto"),
                     primary_idx: Optional[str] = Form(None),
                     transcribe_model: Optional[str] = Form(None),
                     transcribe_lang: str = Form("auto"),
                     transcribe_device: Optional[str] = Form(None),
                     transcribe_compute_type: Optional[str] = Form(None),
                     transcribe_batch_size: Optional[int] = Form(None),
                     screenshots_fps: float = Form(0.0),
                     transcribe_only: bool = Form(False),
                     transcribe: bool = Form(True),
                     reasoning: str = Form(""),
                     reasoning_max_tokens: Optional[int] = Form(None),
                     pdf_as_file: bool = Form(False),
                     pdf_engine: Optional[str] = Form(None)):
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > config.MAX_FILES:
        raise HTTPException(400, f"Too many files (max {config.MAX_FILES}).")
    job_dir = pipeline.new_job_dir()
    total = 0
    saved = 0
    primary_names: list[str] = []
    want_primary: set[int] = set()
    for tok in (primary_idx or "").split(","):
        tok = tok.strip()
        if tok.isdigit():
            want_primary.add(int(tok))
    for i, up in enumerate(files):
        ext = Path(up.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            (job_dir / "job.log").open("a").write(f"Skipped unsupported file: {up.filename}\n")
            continue
        safe = _safe_filename(up.filename or "")
        dest = job_dir / "uploads" / f"{i:02d}_{safe}"
        n = 1
        while dest.exists():
            dest = job_dir / "uploads" / f"{i:02d}_{dest.stem}_{n}{dest.suffix}"
            n += 1
        with open(dest, "wb") as out:
            while chunk := await up.read(1024 * 1024):
                total += len(chunk)
                if total > config.MAX_TOTAL_MB * 1024 * 1024:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(400, "Total upload too large.")
                out.write(chunk)
        saved += 1
        if i in want_primary:
            primary_names.append(dest.name)
    if saved == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "No supported files. Allowed: pdf, images, audio, video, txt/md/srt/vtt/csv.")
    (job_dir / "status.txt").write_text("queued", encoding="utf-8")
    background_tasks.add_task(
        pipeline.run_job, job_dir,
        model=model or config.DEFAULT_MODEL,
        api_key=api_key or "",
        base_url=base_url or config.LLM_BASE,
        temperature=float(temperature),
        language=language,
        primary_names=primary_names,
        transcribe_model=transcribe_model or config.TRANSCRIBE_MODEL,
        transcribe_lang=transcribe_lang,
        transcribe_device=transcribe_device,
        transcribe_compute_type=transcribe_compute_type,
        transcribe_batch_size=transcribe_batch_size,
        screenshots_fps=screenshots_fps or None,
        transcribe_only=transcribe_only,
        reasoning=reasoning or config.REASONING,
        reasoning_max_tokens=reasoning_max_tokens,
        transcribe=transcribe,
        pdf_as_file=pdf_as_file,
        pdf_engine=pdf_engine or config.PDF_ENGINE)
    return {"id": job_dir.name}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_dir = config.DATA_DIR / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found.")
    return JSONResponse(pipeline.job_info(job_dir))


@app.post("/api/jobs/{job_id}/fix")
def fix_job(job_id: str, background_tasks: BackgroundTasks,
            api_key: Optional[str] = Form(None),
            model: Optional[str] = Form(None),
            temperature: Optional[float] = Form(None),
            base_url: Optional[str] = Form(None),
            reasoning: Optional[str] = Form(None),
            reasoning_max_tokens: Optional[int] = Form(None)):
    job_dir = config.DATA_DIR / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found.")
    from .latex_utils import find_tex
    if find_tex(job_dir) is None:
        raise HTTPException(400, "No .tex to fix yet.")
    background_tasks.add_task(pipeline.retry_fix, job_dir,
                              model or None, temperature,
                              api_key or "", base_url or None,
                              reasoning or "", reasoning_max_tokens)
    return {"id": job_id, "status": "fixing"}


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    job_dir = config.DATA_DIR / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found.")
    # Flat names only (no slashes possible in {name}); served from the job
    # root or its transcripts/ subdir. Anything else -> 404.
    safe = Path(name).name
    if not safe or safe != name or safe.startswith("."):
        raise HTTPException(404, "File not found.")
    for cand in (job_dir / safe, job_dir / "transcripts" / safe):
        if cand.is_file():
            media = "application/pdf" if cand.suffix == ".pdf" else "text/plain"
            return FileResponse(str(cand), media_type=media, filename=safe)
    raise HTTPException(404, "File not found.")
