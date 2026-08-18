"""Run Mistral OCR with exercise and instruction metadata."""

import json
import os
from pathlib import Path

from kms.construction import ocr

CASES = (
    ('ea2e_review', 'tests/fixtures/books/ea2e_ch1_review.pdf', [0, 1]),
    (
        'calc3_gradients',
        'tests/fixtures/books/calc3_gradients_exercises.pdf',
        [0, 1],
    ),
    (
        'lebl_analysis',
        'tests/fixtures/books/lebl_realanalysis_sec2_1_exercises.pdf',
        [0, 1],
    ),
)
OUTPUT = Path('output/live_ocr_exercise_metadata')

EXERCISE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'exercise_metadata',
        'schema': {
            'type': 'object',
            'properties': {
                'content_kind': {
                    'type': 'string',
                    'enum': ['instruction', 'exercise', 'mixed', 'other'],
                },
                'exercise_numbers': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'instruction_text': {'type': 'string'},
                'governed_exercise_numbers': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'has_solution_or_worked_answer': {'type': 'boolean'},
            },
            'required': [
                'content_kind',
                'exercise_numbers',
                'instruction_text',
                'governed_exercise_numbers',
                'has_solution_or_worked_answer',
            ],
            'additionalProperties': False,
        },
    },
}

INSTRUCTION_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'exercise_instructions',
        'schema': {
            'type': 'object',
            'properties': {
                'instructions': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
            },
            'required': ['instructions'],
            'additionalProperties': False,
        },
    },
}

PROMPT = (
    'Extract exercise instructions only. Return every complete, standalone '
    'directive that is visibly unnumbered and introduces or governs multiple '
    'exercises or a group of exercise subparts. A qualifying instruction '
    'normally has both a shared-exercise cue ("for the following exercises" '
    'or "in the following exercises") and a complete action, such as '
    '"simplify each expression" or "find the gradient". Do not return a '
    'fragment whose action is unfinished, such as "In the following '
    'exercises, simplify" or "For the following exercises, find equations '
    'of", unless the page immediately continues it with its lettered or '
    'bulleted subparts. When it does, return the lead-in and all of those '
    'continuation lines together as one instruction string. Preserve the '
    'source text exactly as visible, including capitalization, punctuation, '
    'spelling, whitespace, line breaks, bullets, mathematical notation, '
    'Unicode, and OCR oddities. Do not correct duplicated punctuation, '
    'replace symbols, complete fragments, normalize Markdown, or repair OCR. '
    'Never invent a missing character. Exclude every numbered exercise prompt '
    'and every directive belonging to only one numbered exercise, even if it '
    'is imperative. Exclude exercise numbers, answers, solutions, examples, '
    'definitions, captions, headings, learning objectives, explanatory prose, '
    'navigation, and isolated fragments. Do not infer which exercises an '
    'instruction governs and do not return associations, categories, or '
    'metadata. If no qualifying instruction is visible, return an empty list.'
)


def main() -> None:
    """OCR exercise pages and print returned structural metadata."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, pdf_name, pages in CASES:
        options = ocr.OCRRequestOptions(
            include_blocks=True,
            pages=pages,
            document_annotation_format=INSTRUCTION_FORMAT,
            document_annotation_prompt=PROMPT,
        )
        response = ocr.ocr_pdf(Path(pdf_name).read_bytes(), options=options)
        path = OUTPUT / f'{name}.json'
        path.write_text(
            json.dumps(response.raw_response, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'\n=== {name} ===')
        print(f'saved: {path}')
        print(f'{response.document_annotation!r}')


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    main()
