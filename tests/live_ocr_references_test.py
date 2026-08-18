"""Run Mistral OCR with document reference annotation."""

import json
import os
from pathlib import Path

from kms.construction import ocr

PDF = Path('tests/fixtures/books/combinatorics_levin.pdf')
OUTPUT = Path('output/live_ocr_references')
PAGES = [0, 1, 2, 3]


def main() -> None:
    """OCR representative pages and print detected document references."""
    options = ocr.OCRRequestOptions.with_references().model_copy(
        update={'include_blocks': True, 'pages': PAGES}
    )
    response = ocr.ocr_pdf(PDF.read_bytes(), options=options)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    response_path = OUTPUT / 'response.json'
    response_path.write_text(
        json.dumps(response.raw_response, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    print(f'saved: {response_path}')
    print(f'pages: {len(response.pages)}')
    print(f'annotation: {response.document_annotation!r}')


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    main()
