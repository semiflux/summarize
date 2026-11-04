import logging
import re
import subprocess
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


class LatexProcessingError(RuntimeError):
    pass


BABEL_LANGUAGE_MAP = {"it": "italian", "en": "english"}
DEFAULT_BABEL_LANGUAGE = "italian"

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("geometry", "\\usepackage{geometry}"),
    ("graphicx", "\\usepackage{graphicx}"),
    ("hyperref", "\\usepackage{hyperref}"),
    ("amsmath", "\\usepackage{amsmath}"),
    ("amssymb", "\\usepackage{amssymb}"),
    ("enumitem", "\\usepackage{enumitem}"),
    ("caption", "\\usepackage{caption}"),
    ("xcolor", "\\usepackage{xcolor}"),
    ("inputenc", "\\usepackage[utf8]{inputenc}"),
    ("microtype", "\\usepackage{microtype}"),
    ("fontenc", "\\usepackage[T1]{fontenc}"),
    ("csquotes", "\\usepackage{csquotes}"),
)

CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:latex|tex)?\s*\n?|```\s*$", re.IGNORECASE | re.MULTILINE)


def sanitise_latex(raw_output: str, preferred_language: str | None = None) -> str:
    text = CODE_FENCE_PATTERN.sub("", raw_output.strip())
    doc_start = text.find("\\documentclass")
    doc_end = text.rfind("\\end{document}")
    if doc_start != -1 and doc_end != -1:
        text = text[doc_start : doc_end + len("\\end{document}")]
    else:
        logger.warning("Could not find documentclass/end{document}. Using raw output.")
    if "\\documentclass" not in text:
        logger.warning("Missing documentclass; injecting default.")
        text = "\\documentclass[11pt]{article}\n" + text
    text = _ensure_required_packages(text, preferred_language)
    return text.strip()


def _ensure_required_packages(latex: str, preferred_language: str | None = None) -> str:
    m = re.search(r"\\begin\{document\}", latex)
    if not m:
        raise LatexProcessingError("Generated LaTeX is missing \\begin{document}")
    preamble, body = latex[: m.start()], latex[m.start() :]
    for package, snippet in REQUIRED_PACKAGES:
        if not re.search(rf"\\usepackage(?:\[[^\]]*\])?\{{{package}\}}", preamble):
            preamble += f"\n{snippet}"
    if not re.search(r"\\usepackage(?:\[[^\]]*\])?\{babel\}", preamble):
        lang = BABEL_LANGUAGE_MAP.get((preferred_language or "").lower(), DEFAULT_BABEL_LANGUAGE)
        preamble += f"\n\\usepackage[{lang}]{{babel}}"
    return preamble + "\n" + body


def _slugify(raw: str) -> str:
    clean = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^{}]*\})?", "", raw)
    norm = unicodedata.normalize("NFKD", clean).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_")[:60].strip("_")


def title_slug(tex: str, fallback: str = "lecture_notes") -> str:
    """Descriptive ASCII filename stem from \\title{}, else from fallback name."""
    m = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    slug = _slugify(m.group(1)) if m else ""
    return slug or _slugify(fallback) or "lecture_notes"


def find_tex(job_dir: Path) -> Path | None:
    texs = sorted(job_dir.glob("*.tex"))
    return texs[0] if texs else None


def compile_pdf(tex_path: Path) -> Path:
    """Try tectonic -> latexmk -> pdflatex. Raises LatexProcessingError with log excerpts."""
    pdf_path = tex_path.with_suffix(".pdf")
    workdir = tex_path.parent
    commands: tuple[tuple[str, ...], ...] = (
        ("tectonic", tex_path.name),
        ("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name),
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name),
    )
    failures: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                command, cwd=workdir, check=False,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            logger.warning("LaTeX executable '%s' not available.", command[0])
            failures.append(f"[{command[0]}] not installed.")
            continue
        if result.returncode == 0 and pdf_path.exists():
            logger.info("Compiled with '%s' -> %s", command[0], pdf_path)
            return pdf_path
        tail = (result.stdout[-2000:] + "\n" + result.stderr[-2000:]).strip()
        logger.error("'%s' failed (exit %d): %s %s",
                     command[0], result.returncode, result.stdout[-1500:], result.stderr[-1500:])
        failures.append(f"[{command[0]} exit {result.returncode}]\n{tail}")
    raise LatexProcessingError(
        f"All LaTeX compilation attempts failed for {tex_path.name}:\n\n"
        + "\n\n".join(failures)[-6000:]
    )


def create_dark_pdf(pdf_path: Path) -> Path | None:
    dark_pdf_path = pdf_path.with_stem(pdf_path.stem + "_dark")
    command = (
        "gs", "-o", str(dark_pdf_path), "-sDEVICE=pdfwrite",
        "-dProcessColorModel=/DeviceGray", "-sColorConversionStrategy=Gray",
        "-c", "<< /BeginPage { gsave 0.87 setgray clippath fill grestore true } >> setpagedevice {1 exch sub} settransfer",
        "-f", str(pdf_path),
    )
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        logger.warning("Ghostscript (gs) not found. Skipping dark PDF.")
        return None
    if result.returncode == 0 and dark_pdf_path.exists():
        return dark_pdf_path
    logger.error("Ghostscript failed: %s", result.stderr[-1000:])
    return None
