"""Run local Qwen block correction on Mistral OCR crops."""

import asyncio
import json
import os
from pathlib import Path

from kms.construction import block_corrector, ocr
from kms.core import llm

CASES = (
    (
        'logic_hammack_truthtables_p00',
        'tests/fixtures/books/logic_hammack_truthtables.pdf',
        [0],
    ),
    (
        'nt_stein_congruences_p02',
        'tests/fixtures/books/nt_stein_congruences.pdf',
        [2],
    ),
    (
        'topology_morris_p02',
        'tests/fixtures/books/topology_morris.pdf',
        [2],
    ),
)
OUTPUT = Path('output/live_ocr_block_correction')


async def run_case(name: str, pdf_name: str, pages: list[int]) -> None:
    pdf_path = Path(pdf_name)
    response = ocr.ocr_pdf(
        pdf_path.read_bytes(), pages=pages, include_blocks=True
    )
    document_dir = OUTPUT / name
    document = ocr.materialize_document(
        response,
        document_dir,
        pdf_path=pdf_path,
        pages=pages,
    )
    corrector = block_corrector.BlockCorrector(llm.module_lm('corrector'))
    results = []
    for page in document.pages:
        for region in page.blocks:
            if not region.crop_path:
                continue
            correction = await corrector.acorrect(region)
            results.append(
                {
                    'page_index': page.index,
                    'block_index': region.block_index,
                    'block_type': region.block.type,
                    'crop_path': region.crop_path,
                    'crop_bbox': region.crop_bbox,
                    'original_text': region.block.content or '',
                    'corrected_text': correction['corrected_text'],
                    'changes': correction['changes'],
                }
            )
    path = document_dir / 'block_corrections.json'
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'{name}: {len(results)} blocks -> {path}')
    for result in results:
        if result['changes']:
            print(f'  block {result["block_index"]}: {result["changes"]}')


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        await run_case(*case)


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    asyncio.run(main())
