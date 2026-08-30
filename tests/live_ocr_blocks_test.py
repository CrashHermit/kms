"""Inspect Mistral OCR structural blocks on representative correction pages."""

import difflib
import json
import re
from pathlib import Path

from kms.construction import ocr

CASES = (
    {
        'name': 'logic_hammack_truthtables_p00',
        'pdf': 'tests/fixtures/books/logic_hammack_truthtables.pdf',
        'page': 0,
        'transcription': (
            'data/gold/corrector/real/'
            'logic_hammack_truthtables_p00/transcription.md'
        ),
    },
    {
        'name': 'nt_stein_congruences_p02',
        'pdf': 'tests/fixtures/books/nt_stein_congruences.pdf',
        'page': 2,
        'transcription': (
            'data/gold/corrector/real/nt_stein_congruences_p02/transcription.md'
        ),
    },
    {
        'name': 'topology_morris_p02',
        'pdf': 'tests/fixtures/books/topology_morris.pdf',
        'page': 2,
        'transcription': (
            'data/gold/corrector/real/topology_morris_p02/transcription.md'
        ),
    },
)


def _normalise(text: str) -> str:
    """Collapse whitespace for a rough serialization comparison."""
    return re.sub(r'\s+', ' ', text).strip()


def _block_text(block: dict) -> str:
    """Extract printable content from one structural block."""
    content = block.get('content', '')
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _summarise(name: str, page: dict, gold_markdown: str) -> None:
    """Print block counts and rough text-serialization diagnostics."""
    markdown = page.get('markdown', '') or ''
    blocks = page.get('blocks', []) or []
    block_texts = [_block_text(block) for block in blocks]
    joined_blocks = '\n'.join(text for text in block_texts if text)
    markdown_normalised = _normalise(markdown)
    blocks_normalised = _normalise(joined_blocks)
    gold_normalised = _normalise(gold_markdown)
    print(f'\n=== {name} ===')
    print(
        f'markdown: {len(markdown)} chars, {len(markdown.splitlines())} lines'
    )
    print(
        f'gold:     {len(gold_markdown)} chars, {len(gold_markdown.splitlines())} lines'
    )
    print(f'blocks:   {len(blocks)}, {len(joined_blocks)} content chars')
    print(
        'markdown/block normalized similarity: '
        f'{difflib.SequenceMatcher(None, markdown_normalised, blocks_normalised).ratio():.3f}'
    )
    print(
        'gold/block normalized similarity:     '
        f'{difflib.SequenceMatcher(None, gold_normalised, blocks_normalised).ratio():.3f}'
    )
    type_counts: dict[str, int] = {}
    for block in blocks:
        block_type = str(block.get('type', '<missing>'))
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    print(f'types:    {type_counts}')
    for index, block in enumerate(blocks, start=1):
        print(
            f'block {index}: type={block.get("type")!r} '
            f'bbox=({block.get("top_left_x")}, {block.get("top_left_y")}, '
            f'{block.get("bottom_right_x")}, {block.get("bottom_right_y")}) '
            f'chars={len(_block_text(block))}'
        )
        print(_block_text(block)[:500].replace('\n', '\\n'))


def main() -> None:
    """Run the structural-block inspection for the selected fixture pages."""
    output_root = Path('output/live_ocr_blocks')
    output_root.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        response = ocr.ocr_pdf(
            Path(case['pdf']).read_bytes(),
            pages=[case['page']],
            options=ocr.OCRRequestOptions(include_blocks=True),
        )
        response_path = output_root / f'{case["name"]}.json'
        response_path.write_text(
            json.dumps(response, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        page = (response.get('pages') or [{}])[0]
        gold_markdown = Path(case['transcription']).read_text(encoding='utf-8')
        _summarise(case['name'], page, gold_markdown)
        print(f'raw response: {response_path}')


if __name__ == '__main__':
    main()
