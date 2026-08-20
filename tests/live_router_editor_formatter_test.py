"""Run the split formatter router/editor against representative content."""

import asyncio
import json
import os
from pathlib import Path

os.environ['KMS_MODELS__MODULES__FORMATTER__BASE_URL'] = (
    'http://127.0.0.1:8080/v1'
)
os.environ['KMS_MODELS__MODULES__FORMATTER__MODEL'] = 'openai/qwen3.5-9b'
os.environ['KMS_MODELS__MODULES__FORMATTER__API_KEY'] = 'not-needed'

from kms.construction import formatter  # noqa: E402
from kms.core import llm  # noqa: E402
from kms.core.edits import apply_line_edits, number_lines  # noqa: E402


SAMPLE = r"""Quadratic Formula

Given \(ax^2 + bx + c = 0\), the roots are x = (-b ± √(b² - 4ac)) / 2a.

The shirt costs $20, and the total is 3x + 5 = 20.

Solve x² = 9 and x ≤ 5.

Area
====

Find the area where 2 ≤ x ≤ 5 and y = x²."""
OUTPUT = Path('output/live_router_editor_formatter.json')


async def main() -> None:
    """Run one live router/editor pass and save its decisions and edits."""
    module = formatter.Formatter(language_model=llm.module_lm('formatter'))
    lines = number_lines(SAMPLE)
    needs_formatting = await module.router.aforward(lines=lines)
    edits = (
        await module.editor.aforward(lines=lines) if needs_formatting else []
    )
    formatted = apply_line_edits(SAMPLE, edits)
    result = {
        'model': os.environ['KMS_MODELS__MODULES__FORMATTER__MODEL'],
        'needs_formatting': needs_formatting,
        'edits': [edit.model_dump() for edit in edits],
        'input': SAMPLE,
        'output': formatted,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'needs_formatting={needs_formatting}')
    print(f'editor_edits={len(edits)}')
    print(f'saved: {OUTPUT}')
    print('\n=== formatted output ===')
    print(formatted)


if __name__ == '__main__':
    asyncio.run(main())
