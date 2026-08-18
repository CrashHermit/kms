"""Evaluate the visual block router on known OCR perturbations."""

import asyncio
import json
import os
from pathlib import Path

os.environ['KMS_MODELS__MODULES__CORRECTOR__BASE_URL'] = 'http://localhost:8080/v1'
os.environ['KMS_MODELS__MODULES__CORRECTOR__MODEL'] = 'openai/qwen3.5-9b'
os.environ['KMS_MODELS__MODULES__CORRECTOR__API_KEY'] = 'not-needed'

from kms.construction import block_corrector, ocr
from kms.core import llm

CASES = (
    ('logic_hammack_truthtables_p00_v1', 'logic_hammack_truthtables_p00'),
    ('topology_morris_p02_v1', 'topology_morris_p02'),
    ('ode_lebl_diffyqs_p00_v1', 'ode_lebl_diffyqs_p00'),
)
OUTPUT = Path('output/live_ocr_perturbed_block_correction')
INDEX = Path('data/gold/corrector/index.json')


def perturb_region(region, edits: list[dict]) -> list[dict]:
    """Inject the indexed gold errors into the matching OCR block."""
    text = region.block.content or ''
    applied = []
    for edit in edits:
        before = edit['before']
        after = edit['after']
        if after not in text:
            continue
        text = text.replace(after, before, 1)
        applied.append(edit)
    region.block.content = text
    return applied


async def run_case(record: dict, output_name: str) -> None:
    pdf_path = Path(record['source_pdf'])
    response = ocr.ocr_pdf(
        pdf_path.read_bytes(), pages=[record['page']], include_blocks=True
    )
    document_dir = OUTPUT / output_name
    document = ocr.materialize_document(
        response, document_dir, pdf_path=pdf_path, pages=[record['page']]
    )
    edits = record['edits']
    injected = []
    for page in document.pages:
        for region in page.blocks:
            if not region.crop_path:
                continue
            for edit in perturb_region(region, edits):
                injected.append((region.block_index, edit))

    corrector = block_corrector.BlockCorrector(llm.module_lm('corrector'))
    results = []
    for page in document.pages:
        for region in page.blocks:
            if not region.crop_path:
                continue
            original_text = region.block.content or ''
            reviewed = await corrector.reviewer.areview(region)
            correction = (
                await corrector.aforward(
                    crop_path=region.crop_path,
                    block_type=region.block.type,
                    original_text=original_text,
                )
                if reviewed
                else {'corrected_text': original_text, 'changes': []}
            )
            results.append(
                {
                    'block_index': region.block_index,
                    'block_type': region.block.type,
                    'reviewed': reviewed,
                    'original_text': original_text,
                    'corrected_text': correction['corrected_text'],
                    'changes': correction['changes'],
                }
            )

    path = document_dir / 'block_corrections.json'
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    corrected = 0
    for block_index, edit in injected:
        result = next(x for x in results if x['block_index'] == block_index)
        if edit['after'] in result['corrected_text']:
            corrected += 1
        print(
            f"{output_name} block {block_index}: "
            f"reviewed={result['reviewed']} "
            f"expected={edit['before']!r} -> {edit['after']!r}"
        )
        print(f"  output: {result['corrected_text']!r}")
    print(
        f'{output_name}: {len(results)} blocks, '
        f'{sum(x["reviewed"] for x in results)} editor calls, '
        f'{corrected}/{len(injected)} injected errors restored -> {path}'
    )
    if len(injected) != len(edits):
        print(f'  WARNING: injected {len(injected)}/{len(edits)} indexed edits')


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {r['id']: r for r in json.loads(INDEX.read_text())['records']}
    for record_id, output_name in CASES:
        await run_case(records[record_id], output_name)


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    asyncio.run(main())
