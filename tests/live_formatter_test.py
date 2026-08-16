import asyncio
import os

os.environ['KMS_MODELS__MODULES__FORMATTER__BASE_URL'] = (
    'http://localhost:8080/v1'
)
os.environ['KMS_MODELS__MODULES__FORMATTER__MODEL'] = (
    'openai/unsloth/gemma-4-e4b-it-GGUF'
)
os.environ['KMS_MODELS__MODULES__FORMATTER__API_KEY'] = 'not-needed'

from kms.construction import formatter
from kms.core import llm

SAMPLE = r"""Quadratic Formula

Given \(ax^2 + bx + c = 0\), the roots are x = (-b ± √(b² - 4ac)) / 2a.

The shirt costs $20, and the total is 3x + 5 = 20.

Solve x² = 9 and x ≤ 5.

The function f(x) = x² is the square, and π ≈ 3.14159.

Area
====

Find the area where 2 ≤ x ≤ 5 and y = x²."""


async def main():
    module = formatter.Formatter(language_model=llm.module_lm('formatter'))
    numbered = formatter.number_lines(SAMPLE)
    print('=== numbered input ===')
    print(numbered)

    result = await module.predictor.acall(lines=numbered)
    edits = list(result.edits or [])
    print('\n=== edits ===')
    for edit in edits:
        print(f'  [{edit.index}] -> {edit.replacement!r}')

    output = formatter.apply_line_edits(SAMPLE, edits)
    print('\n=== output ===')
    print(output)
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
