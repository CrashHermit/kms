"""Render selected PDF pages into a Segments tree — a docling bypass for GPU-less envs.

The pipeline's front-end (picture_extractor) uses docling only to render page PNGs and
crop figures; the actual text OCR is a vision LLM reading those PNGs. Where docling/torch
aren't available (e.g. the web container has no GPU), render pages with pypdfium2 into the
same disk layout and run the pipeline with extract_pictures=False. Figure cropping is
skipped, so ![N]() placeholders won't resolve — fine for testing text/entity extraction.

    uv pip install pypdfium2 pillow          # test-only deps, NOT in pyproject
    PYTHONPATH=src uv run python scripts/pdf_to_segments.py book.pdf out/ --pages 190-193

Pages are 0-based PDF page indices. Then run the LLM pipeline on the tree:

    PYTHONPATH=src uv run python -c "import asyncio; from module.pipeline import run; \
        asyncio.run(run('unused', output_dir='out/', extract_pictures=False))"

which writes out/document.md and out/entities.json. Tip: use pypdfium2's text extraction
(pdf[i].get_textpage().get_text_range()) to scan for cue-rich pages before rendering.
"""

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def parse_pages(spec: str) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            pages.extend(range(int(lo), int(hi) + 1))
        else:
            pages.append(int(part))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="source PDF path")
    ap.add_argument("out", help="output dir (a Segments/ tree is created under it)")
    ap.add_argument("--pages", required=True, help="0-based pages, e.g. 190-193 or 10,12,15")
    ap.add_argument("--scale", type=float, default=2.5, help="render scale (higher = sharper)")
    args = ap.parse_args()

    pdf = pdfium.PdfDocument(args.pdf)
    for seg_index, page in enumerate(parse_pages(args.pages)):
        seg_dir = Path(args.out) / "Segments" / f"Segment_{seg_index:04d}"
        (seg_dir / "Images").mkdir(parents=True, exist_ok=True)
        pdf[page].render(scale=args.scale).to_pil().save(seg_dir / "Segment.png")
        print(f"Segment_{seg_index:04d} <- pdf page {page}")
    print(f"tree: {args.out}/Segments  (run the pipeline with extract_pictures=False)")


if __name__ == "__main__":
    main()
