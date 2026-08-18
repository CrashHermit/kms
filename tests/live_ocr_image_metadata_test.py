"""Run Mistral OCR with block and image metadata annotations."""

import json
import os
from pathlib import Path

from kms.construction import ocr

PDF = Path('tests/fixtures/books/combinatorics_levin.pdf')
OUTPUT = Path('output/live_ocr_image_metadata')


def main() -> None:
    """OCR two representative pages and print returned image metadata."""
    options = ocr.OCRRequestOptions.with_image_metadata().model_copy(
        update={'include_blocks': True, 'pages': [0, 1]}
    )
    response = ocr.ocr_pdf(PDF.read_bytes(), options=options)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    response_path = OUTPUT / 'response.json'
    response_path.write_text(
        json.dumps(response.raw_response, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    print(f'saved: {response_path}')
    for page in response.pages:
        print(f'\n=== page {page.index} ===')
        print(f'blocks: {len(page.blocks)}')
        print(f'images: {len(page.images)}')
        for index, image in enumerate(page.images, start=1):
            print(f'  image {index}: id={image.id!r}')
            print(f'    annotation={image.image_annotation!r}')


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    main()
