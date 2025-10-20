import os
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from google import genai
from google.genai import types

# Ho corretto l'importazione di PIL (assumendo che fosse 'PIL.Image' o 'Image' come convenzione)
# Visto che il codice originale usava 'pil' e 'image', ho messo 'Image' per coerenza
# Se il tuo ambiente usa 'pil', potresti dover cambiare 'Image' in 'pil.Image'
# Non posso rimuovere i riferimenti a PIL/Image perché la conversione in JPEG è essenziale
from PIL import Image

system_prompt = """you are a latex compositor for university lectures. using the provided input data, produce a single self-contained latex document in italian or english depending on the source content.
requirements:
- output a complete compilable latex file from \\documentclass to \\end{document}.
- use \\documentclass[11pt]{article} and ensure the preamble defines the packages: geometry, graphicx, hyperref, amsmath, amssymb, enumitem, caption, xcolor, babel (with the correct language option), inputenc if utf8 encoding is used, microtype, and csquotes.
- add any additional packages required by features you use (e.g. \\usepackage{siunitx} when using \\si commands).
- title: infer from transcript or filename; include date if present; omit author if unknown.
- language: infer from input data.
- structure the lecture with \\section and \\subsection commands; do not invent topics missing from the transcript.
- include mathematical expressions using display math or inline math when appropriate. avoid figures, images, \\includegraphics, or floats.
- escape special characters and avoid markdown syntax, todos, or placeholder text.
- do not append the raw transcript or verbatim appendix.
"""

def _encode_to_jpeg_bytes(image_data: bytes, quality: int, subsampling: int) -> bytes:
    """Converts raw image bytes to JPEG bytes, handling transparency and mode conversion."""
    buf = BytesIO(image_data)
    with Image.open(buf) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        
        out_buf = BytesIO()
        im.save(out_buf, format="jpeg", quality=quality, optimize=True, subsampling=subsampling)
        return out_buf.getvalue()

def _build_contents(transcript_text: str, image_bytes_list: List[bytes],
                    jpeg_quality: int, jpeg_subsampling: int):
    parts: List[types.Part] = []

    if transcript_text.strip():
        parts.append(types.Part.from_text(text="transcript follows."))
        parts.append(types.Part.from_text(text=transcript_text))

    for i, data in enumerate(image_bytes_list):
        jpeg_data = _encode_to_jpeg_bytes(data, jpeg_quality, jpeg_subsampling)
        
        parts.append(types.Part.from_text(text=f"slide image: {i+1}"))
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=jpeg_data))

    if not parts:
        parts.append(types.Part.from_text(text="no transcript or images provided."))

    return [types.Content(role="user", parts=parts)]

def generate_latex(
    text: Optional[str] = None,
    images: Optional[List[bytes]] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    jpeg_quality: int = 92,
    jpeg_subsampling: int = 0,
    api_key: Optional[str] = None,
) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("environment variable GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    text = text or ""
    images = images or []

    client = genai.Client(api_key=api_key)

    prompt_parts = _build_contents(text, images, jpeg_quality, jpeg_subsampling)

    response = client.models.generate_content(
        model=model,
        contents=prompt_parts,
        config=types.GenerateContentConfig(
            temperature = temperature,
            system_instruction=system_prompt
        )
    )

    return response.text or ""
