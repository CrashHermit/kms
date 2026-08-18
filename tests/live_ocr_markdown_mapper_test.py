"""Run local Qwen block correction and Markdown mapping."""

import asyncio
import json
import os
from pathlib import Path

os.environ['KMS_MODELS__MODULES__CORRECTOR__BASE_URL'] = (
    'http://localhost:8080/v1'
)
os.environ['KMS_MODELS__MODULES__CORRECTOR__MODEL'] = 'openai/qwen3.5-9b'
os.environ['KMS_MODELS__MODULES__CORRECTOR__API_KEY'] = 'not-needed'

from kms.construction import block_corrector, markdown_mapper, ocr
from kms.core import llm

PDF = Path('tests/fixtures/books/nt_stein_congruences.pdf')
PAGES = [2]
OUTPUT = Path('output/live_ocr_markdown_mapper')


async def main() -> None:
    response = ocr.ocr_pdf(PDF.read_bytes(), pages=PAGES, include_blocks=True)
    document_dir = OUTPUT / 'nt_stein_congruences_p02'
    document = ocr.materialize_document(
        response,
        document_dir,
        pdf_path=PDF,
        pages=PAGES,
    )
    corrector = block_corrector.BlockCorrector(llm.module_lm('corrector'))
    mapper = markdown_mapper.MarkdownMapper(llm.module_lm('corrector'))
    corrections = []
    page = document.pages[0]
    for region in page.blocks:
        if not region.crop_path:
            continue
        correction = await corrector.acorrect(region)
        corrections.append(
            markdown_mapper.BlockCorrection(
                block_index=region.block_index,
                block_type=region.block.type,
                original_text=region.block.content or '',
                corrected_text=correction['corrected_text'],
            )
        )
    corrected_markdown = await markdown_mapper.map_and_apply(
        mapper, page.markdown, corrections
    )
    result = {
        'original_markdown': page.markdown,
        'corrected_markdown': corrected_markdown,
        'corrections': [item.model_dump() for item in corrections],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = document_dir / 'mapped_result.json'
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'saved: {path}')


if __name__ == '__main__':
    if not os.environ.get('KMS_OCR__API_KEY'):
        raise RuntimeError('KMS_OCR__API_KEY must be set for this live test.')
    asyncio.run(main())
