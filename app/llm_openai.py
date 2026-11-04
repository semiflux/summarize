"""OpenAI-compatible backend (OpenRouter, Ollama, …): text + images in, LaTeX out.

No video/audio/PDF uploads here — callers send transcripts, texts and
page/frame images. Auth is per-call api_key or LLM_API_KEY env.
The key is never logged.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import time
from pathlib import Path
from typing import Callable, Optional

from . import config
from .prompts import FIX_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

# Paid-tier rates per 1M tokens (input, output). Ballpark for cost display.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "qwen/qwen3.8-max": (2.00, 6.00),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "z-ai/glm-5.3-flash": (0.026, 0.90),
    "z-ai/glm-5.3": (0.079, 4.80),
    "deepseek/deepseek-v4-flash": (0.091, 0.182),
    "deepseek/deepseek-v4.1-flash": (0.044, 0.63),
    "deepseek/deepseek-v4-pro": (0.42, 0.84),
    "mistralai/mistral-large-4-0": (0.68, 2.09),
    "moonshotai/kimi-k2.6": (0.65, 3.41),
    "moonshotai/kimi-k3": (0.73, 10.00),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "thinkingmachines/inkling": (1.00, 4.05),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "google/gemini-3.5-flash": (1.50, 9.00),
}


_PRICE_CACHE: dict[str, tuple[float, float]] = {}
_PRICE_FETCHED = False


def _fetch_openrouter_prices() -> None:
    """One-shot fetch of live per-token prices (USD per 1M) for unknown models."""
    global _PRICE_FETCHED
    if _PRICE_FETCHED:
        return
    _PRICE_FETCHED = True
    try:
        import json as _json
        import urllib.request
        url = (config.LLM_BASE or "").rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.load(r)
        for m in data.get("data", []) or []:
            p = m.get("pricing") or {}
            try:
                pin = float(p.get("prompt", 0)) * 1e6
                pout = float(p.get("completion", 0)) * 1e6
            except (TypeError, ValueError):
                continue
            _PRICE_CACHE[str(m.get("id"))] = (pin, pout)
    except Exception:
        pass


def estimate_cost(usage: dict, model: str) -> dict:
    key = model or ""
    live = False
    rates = MODEL_PRICES.get(key)
    if rates is None and key.endswith(":free"):
        rates = (0.0, 0.0)
    if rates is None and key:
        _fetch_openrouter_prices()
        rates = _PRICE_CACHE.get(key)
        live = rates is not None
    prompt = int(usage.get("prompt", 0))
    output = int(usage.get("output", 0))
    if not rates:
        return {"input_usd": None, "output_usd": None, "total_usd": None,
                "note": f"no rate table for {model}"}
    rin, rout = rates
    total = prompt / 1e6 * rin + output / 1e6 * rout
    if not total:
        note = "free model"
    elif live:
        note = "estimate at live OpenRouter rates"
    else:
        note = "estimate at listed OpenRouter rates"
    return {"input_usd": round(prompt / 1e6 * rin, 4),
            "output_usd": round(output / 1e6 * rout, 4),
            "total_usd": round(total, 4),
            "note": note}

IMAGE_MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}
PDF_MIME = "application/pdf"


def _client(api_key: str = "", base_url: str | None = None):
    from openai import OpenAI
    key = (api_key or "").strip() or config.LLM_API_KEY
    if not key:
        raise ValueError("No LLM API key. Set LLM_API_KEY or pass one per job.")
    return OpenAI(
        api_key=key,
        base_url=(base_url or "").strip() or config.LLM_BASE,
        default_headers={"HTTP-Referer": "https://localhost/",
                         "X-Title": "lesson-summarizer"},
    )


def _usage_dict(usage, finish_reason: str | None = None) -> dict:
    if usage is None:
        return {"prompt": 0, "output": 0, "thoughts": 0, "total": 0, "by_modality": {},
                "finish_reason": finish_reason}
    p = int(getattr(usage, "prompt_tokens", 0) or 0)
    o = int(getattr(usage, "completion_tokens", 0) or 0)
    # Reasoning tokens are billed as output; completion_tokens already counts
    # them, so `thoughts` is informational only (no double counting).
    det = getattr(usage, "completion_tokens_details", None)
    thoughts = int(getattr(det, "reasoning_tokens", 0) or 0) if det else 0
    return {"prompt": p, "output": o, "thoughts": thoughts, "total": p + o,
            "by_modality": {}, "finish_reason": finish_reason}


# Token budget per reasoning level. OpenRouter accepts either `effort` or
# `max_tokens` (never both) and maps `max_tokens` to an effort level on models
# that only support effort, so a budget is the portable, hard-bounding knob.
LEVEL_BUDGETS = {"minimal": 2048, "low": 4096, "medium": 16384, "high": 65536}
DEFAULT_BUDGET = 8192  # used when reasoning is on but no level is picked
_OFF = {"off", "none", "disabled", "false", "0"}


def _reasoning_body(level: str, max_tokens: int = 0) -> dict | None:
    """Map a reasoning level + token ceiling to OpenRouter's `reasoning` object.

    off/none/0 -> {"enabled": false}.
    With a ceiling (the default, see LLM_REASONING_MAX_TOKENS): the level picks a
    token budget, clamped to the ceiling, and we send {"max_tokens": budget}. This
    hard-bounds runaway thinking, which can otherwise spend the entire output
    budget and return nothing.
    With no ceiling (--reasoning-max-tokens 0): fall back to raw effort levels /
    an explicit numeric level, or omit the param entirely (provider default).
    """
    lvl = (level or "").strip().lower()
    if lvl in _OFF:
        return {"enabled": False}
    cap = int(max_tokens or 0)
    if cap <= 0:
        if lvl.isdigit():
            return {"max_tokens": int(lvl)}
        if lvl and lvl not in {"default", "auto", "provider"}:
            return {"effort": lvl}
        return None
    return {"max_tokens": min(LEVEL_BUDGETS.get(lvl, DEFAULT_BUDGET), cap)}


def _reasoning_desc(level: str, max_tokens: int = 0) -> str:
    """Human-readable description of the reasoning setting, for logs."""
    body = _reasoning_body(level, max_tokens)
    if not body:
        return ""
    if body.get("enabled") is False:
        return ", reasoning off"
    if "max_tokens" in body:
        return f", reasoning {level or 'default'} ({body['max_tokens']} tok cap)"
    return f", reasoning {body.get('effort', level)}"


def _reasoning_text(resp) -> str:
    """Reasoning trace from an OpenRouter response, if any (best effort)."""
    try:
        msg = resp.choices[0].message
    except Exception:
        return ""
    parts: list[str] = []
    details = getattr(msg, "reasoning_details", None)
    if details:
        for d in details:
            if isinstance(d, dict):
                t = d.get("text") or d.get("summary")
            else:
                t = getattr(d, "text", None) or getattr(d, "summary", None)
            if t:
                parts.append(str(t))
    if not parts:
        r = getattr(msg, "reasoning", None)
        if r:
            parts.append(str(r))
    return "\n\n".join(p for p in parts if p).strip()


def _chat(client, model: str, system: str, content: list,
          temperature: float, log: LogFn | None, max_wait_s: int = 1200,
          reasoning: str = "", reasoning_max_tokens: int = 0,
          trace: list[str] | None = None,
          plugins: list | None = None):
    """chat.completions with retries on rate-limit/overload errors.

    `reasoning`/`reasoning_max_tokens` select OpenRouter's reasoning budget;
    `plugins` carries OpenRouter-only request extras (e.g. the PDF file-parser);
    `trace`, when given, collects the model's reasoning text for the caller.
    """
    import random
    import re as _re
    waited = 0.0
    attempt = 0
    extra_body: dict = {}
    body = _reasoning_body(reasoning, reasoning_max_tokens)
    if body:
        extra_body["reasoning"] = body
    if plugins:
        extra_body["plugins"] = plugins
    extra = {"extra_body": extra_body} if extra_body else {}
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": content}],
                temperature=temperature, **extra)
            # OpenRouter reports upstream/provider failures as HTTP 200 with an
            # `error` body and null choices; surface it instead of a silent empty.
            err = getattr(resp, "error", None)
            if err or not getattr(resp, "choices", None):
                raise RuntimeError(f"provider returned no choices: {err}")
            if trace is not None:
                txt = _reasoning_text(resp)
                if txt:
                    trace.append(txt)
            return resp
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            # Some providers reject the `reasoning` param outright. Drop just it
            # and retry once so an unsupported model never breaks a working job.
            if "reasoning" in extra_body and "reasoning" in low and any(
                    k in low for k in ("not support", "unsupported", "invalid",
                                       "unknown", "unrecognized")):
                if log:
                    log(f"Provider rejected the reasoning param "
                        f"({msg[:150]}); retrying without it.")
                extra_body.pop("reasoning", None)
                extra = {"extra_body": extra_body} if extra_body else {}
                continue
            retryable = ("429" in msg or "RateLimit" in type(e).__name__
                         or "RESOURCE_EXHAUSTED" in msg or "overloaded" in msg.lower()
                         or "503" in msg or "timeout" in msg.lower())
            m = _re.search(r"retry in ([\d.]+)s|retry_delay\D*([\d.]+)|try again in ([\d.]+)", msg)
            grp = next((g for g in (m.groups() if m else []) if g), None)
            delay = min(float(grp) if grp else 60 * (2 ** attempt), 300)
            if not retryable or waited + delay > max_wait_s:
                raise
            delay += random.uniform(0, 5)
            attempt += 1
            waited += delay
            if log:
                log(f"Rate limited ({msg[:150]}). Waiting {delay:.0f}s… "
                    f"(waited {waited:.0f}s total)")
            time.sleep(delay)


def _text_block(s: str) -> dict:
    return {"type": "text", "text": s}


def _image_block(label: str, data: bytes, suffix: str) -> list[dict]:
    mime = IMAGE_MIMES.get(suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(data).decode()
    return [_text_block(f"screenshot/slide image: {label}"),
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}}]


def _file_block(label: str, filename: str, data: bytes) -> list[dict]:
    """One document as a provider-parsed `file` part (OpenAI/OpenRouter shape).

    The provider does its own text/image extraction, so no local rendering.
    """
    b64 = base64.b64encode(data).decode()
    return [_text_block(f"pdf document: {label}"),
            {"type": "file",
             "file": {"filename": filename,
                      "file_data": f"data:{PDF_MIME};base64,{b64}"}}]


def generate_notes(system: str, texts: list[tuple[str, str]],
                   images: list[tuple[str, bytes, str]],
                   model: Optional[str] = None, api_key: str = "",
                   base_url: str | None = None, temperature: float = 0.2,
                   log: LogFn | None = None, reasoning: str = "",
                   reasoning_max_tokens: int = 0,
                   trace: list[str] | None = None,
                   documents: list[tuple[str, str, bytes]] | None = None,
                   pdf_engine: str = "") -> tuple[str, dict]:
    model = model or config.LLM_MODEL
    client = _client(api_key, base_url)
    content: list[dict] = [_text_block("transcripts / text files follow.")]
    for name, body in texts:
        if body.strip():
            content.append(_text_block(f"===== transcript: {name} =====\n{body}"))
    for label, data, suffix in images:
        content.extend(_image_block(label, data, suffix))
    documents = documents or []
    for label, filename, data in documents:
        content.extend(_file_block(label, filename, data))
    # OpenRouter-only: pick the PDF parsing engine (ignored when no PDF is sent
    # whole). `auto` leaves the choice to OpenRouter, so no plugin is sent.
    engine = (pdf_engine or "").strip()
    plugins = ([{"id": "file-parser", "pdf": {"engine": engine}}]
               if documents and engine and engine != "auto" else None)
    if log:
        n_img = len(images)
        n_txt = sum(len(b) for _, b in texts)
        extra = f"{n_img} images" + (f" + {len(documents)} pdf file(s)"
                                     if documents else "")
        eng = f" [pdf engine {engine}]" if plugins else ""
        log(f"Generating notes via OpenAI-compatible {model} "
            f"({n_txt} text chars, {extra}){eng}"
            f"{_reasoning_desc(reasoning, reasoning_max_tokens)}…")
    resp = _chat(client, model, system, content, temperature, log,
                 reasoning=reasoning, reasoning_max_tokens=reasoning_max_tokens,
                 trace=trace, plugins=plugins)
    text = ""
    fr = None
    try:
        text = resp.choices[0].message.content or ""
        fr = resp.choices[0].finish_reason
    except Exception:
        pass
    if log and fr and fr != "stop":
        log(f"Response finish_reason={fr}")
    return text, _usage_dict(getattr(resp, "usage", None), fr)


def fix_latex(broken_tex: str, errors: str, model: Optional[str] = None,
              api_key: str = "", base_url: str | None = None,
              temperature: float = 0.2,
              log: LogFn | None = None, reasoning: str = "",
              reasoning_max_tokens: int = 0,
              trace: list[str] | None = None) -> tuple[str, dict]:
    from .prompts import FIX_SYSTEM_PROMPT
    model = model or config.LLM_MODEL
    client = _client(api_key, base_url)
    content = [_text_block("broken latex follows."),
               _text_block(broken_tex[-60000:]),
               _text_block("compiler errors follow."),
               _text_block(errors[-8000:])]
    resp = _chat(client, model, FIX_SYSTEM_PROMPT, content, temperature, log,
                 reasoning=reasoning, reasoning_max_tokens=reasoning_max_tokens,
                 trace=trace)
    text = ""
    fr = None
    try:
        text = resp.choices[0].message.content or ""
        fr = resp.choices[0].finish_reason
    except Exception:
        pass
    if log and fr and fr != "stop":
        log(f"Response finish_reason={fr}")
    return text, _usage_dict(getattr(resp, "usage", None), fr)


def collect_texts_images(files: list[Path], exclude: set[str],
                         extra_texts: list[tuple[str, str]], job_dir: Path,
                         log: LogFn | None = None, as_file: bool = False
                         ) -> tuple[list[tuple[str, str]],
                                    list[tuple[str, bytes, str]],
                                    list[tuple[str, str, bytes]]]:
    """Split job files into (texts, images, documents) for chat APIs.

    PDFs are rendered to page images (PyMuPDF) by default; with `as_file` each
    PDF is sent whole as a `file` part for the provider to parse instead (see
    config.PDF_AS_FILE). Audio/video originals are expected to be covered by
    transcripts (see exclude); anything left over raises a clear error instead
    of being silently dropped.
    """
    from . import media_prep
    texts: list[tuple[str, str]] = list(extra_texts or [])
    images: list[tuple[str, bytes, str]] = []
    documents: list[tuple[str, str, bytes]] = []

    def render_pages(path: Path) -> None:
        pages = media_prep.pdf_pages(path, job_dir / "pdf_pages", log=log)
        for i, pg in enumerate(pages):
            images.append((f"{path.name} (page {i + 1}/{len(pages)})",
                           pg.read_bytes(), pg.suffix.lower()))

    for path in files:
        if path.name in exclude:
            continue
        ext = path.suffix.lower()
        if ext in config.TEXT_EXTS:
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                if log:
                    log(f"Skipping unreadable text file {path.name}: {e}")
                continue
            if body.strip():
                texts.append((path.name, body))
        elif ext in config.IMAGE_EXTS:
            if path.stat().st_size > config.MAX_IMAGE_MB * 1024 * 1024:
                if log:
                    log(f"Skipping oversized image {path.name} "
                        f"({path.stat().st_size / 1e6:.0f} MB > {config.MAX_IMAGE_MB} MB)")
                continue
            images.append((path.name, path.read_bytes(), ext))
        elif ext in config.PDF_EXTS:
            size = path.stat().st_size
            if as_file and size <= config.PDF_MAX_MB * 1024 * 1024:
                if log:
                    log(f"Sending {path.name} as a file part "
                        f"({size / 1e6:.1f} MB, provider-parsed)…")
                documents.append((path.name, path.name, path.read_bytes()))
            elif as_file:
                if log:
                    log(f"{path.name} is {size / 1e6:.1f} MB (> {config.PDF_MAX_MB} MB "
                        f"file-input cap) — rendering pages instead")
                render_pages(path)
            else:
                render_pages(path)
        elif ext in config.AUDIO_EXTS or ext in config.VIDEO_EXTS:
            raise ValueError(
                f"'{path.name}' needs local transcription first: this backend "
                f"takes text+images only. Enable the transcribe toggle.")
        else:
            if log:
                log(f"Skipping unsupported file for this backend: {path.name}")
    if not texts and not images and not documents:
        raise ValueError("Nothing usable for this backend (need texts/transcripts/images).")
    return texts, images, documents
