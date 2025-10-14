import fitz
from pathlib import Path

def render_pdf_to_images(
    input_path: Path,
    output_path: Path,
    dpi: int = 150,
    start_index: int = 1,
    page_from: int | None = None,
    page_to: int | None = None
) -> int:
    doc = fitz.open(str(input_path))
    count = 0
    total_pages = len(doc)

    start_page = page_from - 1 if page_from else 0
    end_page = page_to if page_to else total_pages

    for i, page_number in enumerate(range(start_page, end_page), start=start_index):
        page = doc[page_number]
        filename = f"page{i:04d}_{input_path.stem}.png"
        page.get_pixmap(dpi=dpi, alpha=False).save(str(output_path / filename))
        count += 1

    doc.close()
    return count
