"""
Generate corrector training data from Mistral OCR.

1. Mistral OCR a few pages -> transcription
2. Render page images (pypdfium2)
3. qwen3.7-plus (teacher) corrects each page -> gold output
4. Record (image, flawed transcription, corrected) triples

Produces: output/examples/corrector.json

Usage:
    python -m kms.training.generate_corrector_data tests/fixtures/books/calc3_gradients_exercises.pdf
"""

import base64
import sys
from pathlib import Path

import dspy

from kms.core import llm
from kms.core.llm import OPENROUTER_ENV_KEY
from kms.core.recorder import record_example
from kms.ingestion import corrector, ocr


def _teacher_lm() -> dspy.LM:
    return dspy.LM(
        'openrouter/qwen/qwen3.7-plus',
        api_key=llm._require_key(OPENROUTER_ENV_KEY, 'sk-or-...'),
        temperature=0.0,
        max_tokens=128000,
    )


def _image_for_page(image_path: str) -> dspy.Image:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def generate(pdf_path: str, num_pages: int = 4) -> None:
    teacher_lm = _teacher_lm()
    teacher = corrector.Corrector(language_model=teacher_lm)

    # 1. Mistral OCR + page renders
    pages = list(range(num_pages))
    segments = ocr.extract(pdf_path, output_dir='output', pages=pages)

    # 2. Teacher correction pass
    for segment in segments:
        if not segment.content or not segment.image_path:
            continue

        image = _image_for_page(segment.image_path)
        transcription = segment.content

        print(f'  page {segment.index}: correcting...')
        gold = teacher.forward(page_image=image, transcription=transcription)
        print(
            f'  page {segment.index}: '
            f'{len(transcription)} chars in, {len(gold)} chars out'
        )

        record_example(
            'corrector',
            {'page_image': image, 'transcription': transcription},
            {'corrected': gold},
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(
            'Usage: python -m kms.training.generate_corrector_data'
            ' <pdf_path> [num_pages]'
        )
        sys.exit(1)

    pdf_path = sys.argv[1]
    num_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    generate(pdf_path, num_pages)


if __name__ == '__main__':
    main()
