"""Standalone HTTP transcription service (faster-whisper).

    uvicorn app.main:app --host 127.0.0.1 --port 8100    # or: python -m app.main

API
---
GET  /                         minimal drag-and-drop page
GET  /health                   engine + queue status
POST /transcribe               sync:  multipart `file` -> JSON (blocks until done)
POST /jobs                     async: multipart `file` -> {"id": ...}
GET  /jobs                     recent jobs
GET  /jobs/{id}                status + log + metadata
GET  /jobs/{id}/transcript     the transcript as text/plain

There is no authentication: bind it to a trusted address and gate access
yourself (see README). Uploads live under DATA_DIR/jobs and are pruned after
JOB_TTL_HOURS. Transcriptions are serialised, so concurrent requests queue.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import config, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("transcriber")

app = FastAPI(title="Transcriber", version="1.0")


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"          # queued | running | done | error
    requested_language: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    language: str = ""
    duration: float = 0.0
    segments: int = 0
    chars: int = 0
    error: str = ""
    log: list[str] = field(default_factory=list)


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_QUEUE: "queue.Queue[str]" = queue.Queue()


def _job_dir(job_id: str) -> Path:
    return config.DATA_DIR / "jobs" / job_id


def _mk_log(job: Job):
    def _log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        logger.info("[%s] %s", job.id, msg)
        with _LOCK:
            job.log.append(line)
            del job.log[:-200]      # keep only the tail
    return _log


def _prune() -> None:
    """Drop job dirs (and their registry entries) older than JOB_TTL_HOURS."""
    if config.JOB_TTL_HOURS <= 0:
        return
    root = config.DATA_DIR / "jobs"
    if not root.exists():
        return
    cutoff = time.time() - config.JOB_TTL_HOURS * 3600
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                with _LOCK:
                    _JOBS.pop(d.name, None)
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _read_upload(job_id: str, up: UploadFile) -> Path:
    """Stream an upload to DATA_DIR/jobs/<id>/input<ext>, enforcing the cap."""
    ext = Path(up.filename or "").suffix.lower()
    if ext not in config.MEDIA_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext or up.filename}'. "
                                 f"Allowed: {', '.join(sorted(config.MEDIA_EXTS))}")
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"input{ext}"
    limit = config.MAX_FILE_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as out:
        while chunk := up.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"File too large (max {config.MAX_FILE_MB} MB).")
            out.write(chunk)
    if size == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Empty upload.")
    return dest


def _run(job: Job, src: Path) -> None:
    job.status = "running"
    job.started = time.time()
    try:
        res = engine.transcribe_file(src, language=job.requested_language or None,
                                     log=_mk_log(job))
        job.language = res.language
        job.duration = res.duration
        job.segments = res.segments
        job.chars = len(res.text)
        (src.parent / "transcript.txt").write_text(res.text, encoding="utf-8")
        job.status = "done"
    except Exception as e:
        job.error = str(e)
        job.status = "error"
        _mk_log(job)(f"ERROR: {e}")
    finally:
        job.finished = time.time()


def _worker() -> None:
    """Single worker: jobs run one at a time, in submission order."""
    while True:
        job_id = _QUEUE.get()
        try:
            with _LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                continue
            src = next(iter(sorted(_job_dir(job_id).glob("input.*"))), None)
            if src is None:
                job.status, job.error = "error", "upload missing"
                continue
            _run(job, src)
        except Exception as e:              # never let the worker die
            logger.exception("worker error on %s: %s", job_id, e)
        finally:
            _QUEUE.task_done()


threading.Thread(target=_worker, name="transcriber-worker", daemon=True).start()


@app.get("/health")
def health():
    dev, ct = engine.resolve()
    return {"status": "ok",
            "engine_available": engine.available(),
            "model": config.MODEL,
            "device": dev,
            "compute_type": ct,
            "batch_size": config.BATCH_SIZE,
            "gpu": engine.gpu_available(),
            "queue": _QUEUE.qsize(),
            "jobs_known": len(_JOBS)}


@app.post("/jobs")
def create_job(file: UploadFile = File(...), language: str = Form("")):
    """Queue a transcription; poll GET /jobs/{id}, then GET .../transcript."""
    _prune()
    jid = uuid.uuid4().hex[:12]
    _read_upload(jid, file)
    job = Job(id=jid, name=Path(file.filename or "audio").name,
              requested_language=(language or "").strip())
    with _LOCK:
        _JOBS[jid] = job
    _QUEUE.put(jid)
    return {"id": jid, "status": job.status, "queue": _QUEUE.qsize()}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: str = Form(""),
               model: str = Form("")):
    """Transcribe one file and return the text directly (blocks until done).

    Handy for short clips; for long recordings prefer POST /jobs so the request
    does not sit open for minutes.
    """
    jid = "sync-" + uuid.uuid4().hex[:8]
    try:
        src = _read_upload(jid, file)
        res = engine.transcribe_file(
            src, model_name=(model or "").strip() or engine.DEFAULT_MODEL,
            language=(language or "").strip() or None,
            log=lambda m: logger.info("[sync] %s", m))
        return {"text": res.text, "language": res.language,
                "language_probability": round(res.language_probability, 3),
                "duration": round(res.duration, 1), "segments": res.segments,
                "chars": len(res.text)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(_job_dir(jid), ignore_errors=True)


@app.get("/jobs")
def list_jobs(limit: int = 20):
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created,
                      reverse=True)[:max(1, limit)]
        return {"jobs": [{"id": j.id, "name": j.name, "status": j.status,
                          "language": j.language, "duration": j.duration,
                          "chars": j.chars, "created": j.created} for j in jobs]}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (or expired).")
    d = asdict(job)
    d["log"] = job.log[-50:]
    d["elapsed"] = round((job.finished or time.time())
                         - (job.started or job.created), 1)
    return JSONResponse(d)


@app.get("/jobs/{job_id}/transcript")
def job_transcript(job_id: str):
    # The transcript on disk is the source of truth, so it survives a restart.
    f = _job_dir(job_id) / "transcript.txt"
    if f.is_file():
        return PlainTextResponse(f.read_text(encoding="utf-8"))
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (or expired).")
    raise HTTPException(409, f"Job is {job.status}."
                        + (f" {job.error}" if job.error else ""))


PAGE = """<!doctype html>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Transcriber</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:720px;margin:40px auto;padding:0 16px}
input,select,button{width:100%;box-sizing:border-box;background:#1c1c1c;color:#eee;
  border:1px solid #333;border-radius:8px;padding:8px;margin:6px 0}
button{background:#2f6fed;border:0;cursor:pointer;font-size:15px}
pre{background:#0d0d0d;padding:12px;border-radius:8px;white-space:pre-wrap;
  max-height:55vh;overflow:auto}
.hint{color:#888;font-size:13px}
</style>
<h1>Transcriber</h1>
<p class="hint">Drop an audio or video file. It is transcribed on the server —
nothing is sent to a cloud service.</p>
<input id="f" type="file" accept="audio/*,video/*"/>
<select id="lang">
<option value="">language: auto-detect</option>
<option value="it">italiano</option>
<option value="en">english</option>
</select>
<button onclick="go()">Transcribe</button>
<pre id="out">ready.</pre>
<script>
async function go(){
  const f=document.getElementById('f').files[0];
  const out=document.getElementById('out');
  if(!f){out.textContent='pick a file first.';return;}
  out.textContent='uploading '+f.name+'…';
  const fd=new FormData(); fd.append('file',f);
  const lang=document.getElementById('lang').value; if(lang) fd.append('language',lang);
  const r=await fetch('/jobs',{method:'POST',body:fd});
  if(!r.ok){out.textContent='error: '+await r.text();return;}
  const {id}=await r.json();
  for(;;){
    await new Promise(s=>setTimeout(s,2000));
    const j=await (await fetch('/jobs/'+id)).json();
    out.textContent=j.status+'…'+(j.log&&j.log.length?'\\n'+j.log.slice(-3).join('\\n'):'');
    if(j.status==='done'){
      out.textContent=await (await fetch('/jobs/'+id+'/transcript')).text(); return;
    }
    if(j.status==='error'){out.textContent='error: '+j.error; return;}
  }
}
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)
