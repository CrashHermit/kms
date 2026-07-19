"""
Docling picture extractor.

Converts a PDF into per-page renders plus each page's extracted pictures, saved
to disk. This is a stripped-down mimic of Paideia's Docling provider: it keeps
the one piece that actually matters for large documents — batched conversion so
RAM stays bounded — and drops the graph database, config objects, and device
handling.

Why the batching loop is here: with generate_page_images/generate_picture_images
on, docling attaches every rendered page image and every extracted picture onto
the returned document and holds them all in memory until it is freed. Converting
a 1500-page book in one shot would accumulate all of those images at once and
exhaust RAM. Instead we convert in small page windows, save each window's PNGs to
disk, then drop the result and gc() before the next window — so no more than one
window of rendered images is ever resident.

Disk layout mirrors Paideia:
    output/Segments/Segment_0000/Segment.png
    output/Segments/Segment_0000/Images/Image_000.png
    output/Segments/Segment_0000/Images/Image_001.png
    output/Segments/Segment_0001/Segment.png
    ...
"""

import gc
from pathlib import Path

import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument, PictureItem

# --- Tunables ---
IMAGE_SCALE = 3.0   # render resolution; higher = sharper page/picture images
BATCH_SIZE = 10     # pages converted per window — the RAM ceiling knob
DO_OCR = False      # OCR is handled by a separate stage; skip docling's engine
OUTPUT_DIR = Path.cwd() / "output"


def _make_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    options.images_scale = IMAGE_SCALE
    options.generate_page_images = True
    options.generate_picture_images = True
    options.do_ocr = DO_OCR
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )


def _page_pictures(doc: DoclingDocument, page_no: int):
    """Yield the PIL image of every PictureItem on the given page."""
    for item, _ in doc.iterate_items(page_no=page_no):
        if isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image:
                yield image


def _save_page(doc: DoclingDocument, page_no: int, page, output_dir: Path) -> None:
    segment_index = page_no - 1
    segment_dir = output_dir / "Segments" / f"Segment_{segment_index:04d}"
    pictures_dir = segment_dir / "Images"
    pictures_dir.mkdir(parents=True, exist_ok=True)

    page_image = page.image.pil_image
    page_image.save(segment_dir / "Segment.png")
    page_image.close()

    for index, picture in enumerate(_page_pictures(doc, page_no)):
        picture.save(pictures_dir / f"Image_{index:03d}.png")
        picture.close()


def extract(source: str | Path, output_dir: Path = OUTPUT_DIR, batch_size: int = BATCH_SIZE) -> Path:
    """
    Extract per-page renders and their pictures from a PDF, saving PNGs to disk.

    Pages are converted in windows of `batch_size` and each window is freed before
    the next, keeping memory bounded on large documents.

    Returns the output directory.
    """
    source = Path(source).resolve()
    converter = _make_converter()

    pdf = pdfium.PdfDocument(str(source))
    total_pages = len(pdf)
    pdf.close()

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)

        result = converter.convert(source=str(source), page_range=(start, end))
        for page_no, page in result.document.pages.items():
            if start <= page_no <= end:
                _save_page(result.document, page_no, page, output_dir)

        del result
        gc.collect()

    return output_dir


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    out = extract(pdf_path)
    print(f"Extracted page renders and pictures to: {out}")
