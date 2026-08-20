"""Evaluate the visual block router on known OCR perturbations."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from kms.construction import block_corrector, ocr
from kms.core import llm
from kms.core.edits import apply_line_edits

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
    for document_page in document.documents:
        for block_index, node in enumerate(document_page.nodes):
            crop_path = node.provenance.get('crop_path')
            if not crop_path:
                continue
            region = SimpleNamespace(
                crop_path=crop_path,
                block=SimpleNamespace(
                    type=node.provenance.get('provider_type', node.type),
                    content=node.content,
                ),
            )
            for edit in perturb_region(region, edits):
                node.content = region.block.content
                injected.append((block_index, edit))

    corrector = block_corrector.BlockCorrector(llm.module_lm('corrector'))
    results = []
    for document_page in document.documents:
        for block_index, node in enumerate(document_page.nodes):
            crop_path = node.provenance.get('crop_path')
            if not crop_path:
                continue
            original_text = node.content or ''
            region = SimpleNamespace(
                crop_path=crop_path,
                block=SimpleNamespace(
                    type=node.provenance.get('provider_type', node.type),
                    content=original_text,
                ),
            )
            needs_correction = await corrector.router.needs_correction(region)
            edits_for_block = (
                await corrector.editor.aforward(
                    crop_path=crop_path,
                    block_type=region.block.type,
                    original_text=original_text,
                )
                if needs_correction
                else []
            )
            results.append(
                {
                    'block_index': block_index,
                    'block_type': region.block.type,
                    'reviewed': needs_correction,
                    'original_text': original_text,
                    'corrected_text': apply_line_edits(
                        original_text, edits_for_block
                    ),
                    'edits': [edit.model_dump() for edit in edits_for_block],
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
            f'{output_name} block {block_index}: '
            f'reviewed={result["reviewed"]} '
            f'expected={edit["before"]!r} -> {edit["after"]!r}'
        )
        print(f'  output: {result["corrected_text"]!r}')
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
