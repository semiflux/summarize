import os
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from latex_utils import sanitise_latex, compile_pdf

SYSTEM_PROMPT = """You are a LaTeX compositor for university lectures. Using the provided input data, produce a single self-contained LaTeX document in Italian or English depending on the source content.
Requirements:
- Output a complete compilable LaTeX file from \\documentclass to \\end{document}.
- Use \\documentclass[11pt]{article} and ensure the preamble defines the packages: geometry, graphicx, hyperref, amsmath, amssymb, enumitem, caption, xcolor, babel (with the correct language option), inputenc if utf8 encoding is used, microtype, and csquotes.
- Add any additional packages required by features you use (e.g. \\usepackage{siunitx} when using \\si commands).
- Title: infer from transcript or filename; include date if present; omit author if unknown.
- Language: infer from input data.
- Structure the lecture with \\section and \\subsection commands; do not invent topics missing from the transcript.
- Include mathematical expressions using display math or inline math when appropriate. Avoid figures, images, \\includegraphics, or floats.
- Escape special characters and avoid Markdown syntax, TODOs, or placeholder text.
- Do not append the raw transcript or verbatim appendix.
"""

def _encode_as_jpeg_bytes(path: Path, quality: int, subsampling: int) -> bytes:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=subsampling)
        return buf.getvalue()

def _build_contents(transcripts: list[Path], images: list[Path],
                    jpeg_quality: int, jpeg_subsampling: int):
    parts = [{"role": "user", "parts": []}]
    transcript_text = "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in transcripts) if transcripts else ""
    if transcript_text.strip():
        parts[0]["parts"].append({"text": "Transcript follows."})
        parts[0]["parts"].append({"text": transcript_text})
    if images:
        for img in images:
            data = _encode_as_jpeg_bytes(img, jpeg_quality, jpeg_subsampling)
            parts[0]["parts"].append({"text": f"Slide image: {img.name}"})
            parts[0]["parts"].append({"mime_type": "image/jpeg", "data": data})
    if not parts[0]["parts"]:
        parts[0]["parts"].append({"text": "No transcript or images provided."})
    return parts

def generate_latex(
    transcript_files: list[Path] | None = None,
    image_files: list[Path] | None = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    jpeg_quality: int = 92,
    jpeg_subsampling: int = 0,
    api_key: str | None = None,
) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Environment variable GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    transcript_files = transcript_files or []
    image_files = image_files or []

    client = genai.Client(api_key=api_key)

    prompt_parts = _build_contents(transcript_files, image_files, jpeg_quality, jpeg_subsampling)

    response = client.models.generate_content(
        model=model,
        contents=prompt_parts,
        config=types.GenerateContentConfig(
            temperature = temperature,
            system_instruction=SYSTEM_PROMPT
        )
    )

    return response.text or ""
