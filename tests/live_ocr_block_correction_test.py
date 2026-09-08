"""Run local Qwen block correction on Mistral OCR crops."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

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
        pdf_path.read_bytes(),
        pages=pages,
        options=ocr.OCRRequestOptions(include_blocks=True),
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
    for doc in document.documents:
        for region in doc.nodes:
            if not region.provenance.get('crop_path'):
                continue
            region_obj = SimpleNamespace(
                crop_path=region.provenance['crop_path'],
                block=SimpleNamespace(
                    type=region.provenance.get(
                        'provider_type',
                        region.type.value
                        if hasattr(region.type, 'value')
                        else region.type,
                    ),
                    content=region.content or '',
                ),
            )
            correction = await corrector.acorrect(region_obj)
            results.append(
                {
                    'page_index': doc.index,
                    'block_index': region.index,
                    'block_type': region.type.value
                    if hasattr(region.type, 'value')
                    else region.type,
                    'crop_path': region.provenance['crop_path'],
                    'crop_bbox': region.provenance.get('crop_bbox'),
                    'original_text': region.content or '',
                    'corrected_text': correction['corrected_text'],
                    'edits': [
                        edit.model_dump() for edit in correction['edits']
                    ],
                }
            )
    path = document_dir / 'block_corrections.json'
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'{name}: {len(results)} blocks -> {path}')
    for result in results:
        if result['edits']:
            print(f'  block {result["block_index"]}: {result["edits"]}')


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        await run_case(*case)


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    asyncio.run(main())
