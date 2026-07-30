"""
Render the gold set's page images from the committed fixture PDFs.

The page images are the corrector's other input, and they are not committed —
they are reproducible from ``tests/fixtures/books/*.pdf``, which are. This
script rasterizes each page named in ``index.json`` at the scale the OCR
front-end uses (``kms.ingestion.ocr.RENDER_SCALE``), writing one PNG per source
page under the output directory, named as each record's ``page_image``.

Needs the Mistral front-end's extra (pypdfium2):

    uv sync --extra mistral
    python data/gold/corrector/render_pages.py            # -> output/gold_pages
    python data/gold/corrector/render_pages.py <out_dir>
"""

import json
import sys
from pathlib import Path

GOLD = Path(__file__).parent
REPO = GOLD.parents[2]
DEFAULT_OUTPUT = REPO / 'output' / 'gold_pages'


def main() -> None:
    """Render every page the index names, skipping ones already on disk."""
    import pypdfium2 as pdfium

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((GOLD / 'index.json').read_text())
    # Records share pages (a perturbed record reuses its source page's
    # render), so render each (pdf, page) once.
    pages = {
        (record['source_pdf'], record['page'], record['render_scale']): record[
            'page_image'
        ]
        for record in index['records']
    }

    for (pdf_path, page, scale), name in sorted(pages.items()):
        target = output_dir / name
        if target.exists():
            continue
        document = pdfium.PdfDocument(str(REPO / pdf_path))
        try:
            document[page].render(scale=scale).to_pil().save(target)
        finally:
            document.close()
    print(f'{len(pages)} page image(s) in {output_dir}')


if __name__ == '__main__':
    main()
