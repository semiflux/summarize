# latex_utils.py
import logging
import re
import subprocess
from pathlib import Path
from typing import Sequence

# Ottieni un logger specifico per questo modulo
logger = logging.getLogger(__name__)

class LatexProcessingError(RuntimeError):
    """Raised when LaTeX post-processing or compilation fails."""

# Mappa per le lingue di babel
BABEL_LANGUAGE_MAP = {
    "it": "italian",
    "en": "english",
}
DEFAULT_BABEL_LANGUAGE = "italian"

# Pacchetti richiesti che verranno iniettati se mancanti
REQUIRED_PACKAGES: Sequence[tuple[str, str]] = (
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

# Pattern Regex per rimuovere i "code fence" di Markdown
CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:latex|tex)?\s*\n?|```\s*$", re.IGNORECASE | re.MULTILINE)

def sanitise_latex(raw_output: str, preferred_language: str | None = None) -> str:
    """
    Pulisce l'output del modello, rimuove i code fence, isola il documento
    e garantisce che i pacchetti LaTeX richiesti siano presenti.
    """
    # 1. Rimuovi i code fence ```...```
    text = CODE_FENCE_PATTERN.sub("", raw_output.strip())

    # 2. Isola il contenuto del documento da \documentclass a \end{document}
    doc_start = text.find("\\documentclass")
    doc_end = text.rfind("\\end{document}")
    if doc_start != -1 and doc_end != -1:
        text = text[doc_start : doc_end + len("\\end{document}")]
    else:
        logger.warning("Could not find \\documentclass or \\end{document}. Using raw output.")

    # 3. Garantisci la presenza di \documentclass
    if "\\documentclass" not in text:
        logger.warning("Model output missing \\documentclass; injecting default.")
        text = "\\documentclass[11pt]{article}\n" + text

    # 4. Garantisci la presenza dei pacchetti essenziali
    text = _ensure_required_packages(text, preferred_language)
    
    return text.strip()

def _ensure_required_packages(latex: str, preferred_language: str | None = None) -> str:
    """Inietta i pacchetti LaTeX richiesti se mancano nel preambolo."""
    begin_document_match = re.search(r"\\begin\{document\}", latex)
    if not begin_document_match:
        raise LatexProcessingError("Generated LaTeX is missing \\begin{document}")

    preamble = latex[: begin_document_match.start()]
    body = latex[begin_document_match.start() :]

    # Inietta i pacchetti standard
    for package, snippet in REQUIRED_PACKAGES:
        # Pattern per trovare \usepackage{package} o \usepackage[opt]{package}
        pattern = rf"\\usepackage(?:\[[^\]]*\])?\{{{package}\}}"
        if not re.search(pattern, preamble):
            preamble += f"\n{snippet}"
            logger.debug("Injecting missing package: %s", package)
            
    # Gestisci babel
    if not re.search(r"\\usepackage(?:\[[^\]]*\])?\{babel\}", preamble):
        language_key = (preferred_language or "").lower()
        babel_language = BABEL_LANGUAGE_MAP.get(language_key, DEFAULT_BABEL_LANGUAGE)
        preamble += f"\n\\usepackage[{babel_language}]{{babel}}"
        logger.debug("Injecting babel package with language %s", babel_language)

    return preamble + "\n" + body


def compile_pdf(tex_path: Path) -> Path:
    """
    Compila un file LaTeX in PDF, provando diversi compilatori.
    Restituisce il percorso del PDF generato.
    """
    pdf_path = tex_path.with_suffix(".pdf")
    workdir = tex_path.parent
    
    commands: Sequence[Sequence[str]] = (
        ("tectonic", tex_path.name),
        ("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name),
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name),
    )

    for command in commands:
        executable = command[0]
        logger.info("Running '%s' to compile LaTeX...", executable)
        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            logger.warning("LaTeX executable '%s' is not available.", executable)
            continue

        if result.returncode == 0 and pdf_path.exists():
            logger.info("LaTeX compilation succeeded with '%s'. PDF at %s", executable, pdf_path)
            return pdf_path

        # Logga l'errore in modo dettagliato per il debug
        logger.error(
            "LaTeX compilation with '%s' failed (exit code: %d)\n--- STDOUT ---\n%s\n--- STDERR ---\n%s",
            executable,
            result.returncode,
            result.stdout[-1000:],  # Mostra le ultime righe di output
            result.stderr,
        )

    raise LatexProcessingError(f"All LaTeX compilation attempts failed for {tex_path.name}")


def create_dark_pdf(pdf_path: Path) -> Path | None:
    """Crea una variante in modalità scura del PDF usando Ghostscript."""
    dark_pdf_path = pdf_path.with_stem(pdf_path.stem + "_dark")
    command: Sequence[str] = (
        "gs",
        "-o", str(dark_pdf_path),
        "-sDEVICE=pdfwrite",
        "-dProcessColorModel=/DeviceGray",
        "-sColorConversionStrategy=Gray",
        "-c", "<< /BeginPage { gsave 0.87 setgray clippath fill grestore true } >> setpagedevice {1 exch sub} settransfer",
        "-f", str(pdf_path),
    )

    logger.info("Attempting to create dark mode PDF...")
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        logger.warning("Ghostscript (gs) not found. Skipping dark PDF generation.")
        return None

    if result.returncode == 0 and dark_pdf_path.exists():
        logger.info("Created dark PDF at %s", dark_pdf_path)
        return dark_pdf_path

    logger.error(
        "Ghostscript failed (exit code %d) creating dark PDF:\n%s",
        result.returncode, result.stderr
    )
    return None
