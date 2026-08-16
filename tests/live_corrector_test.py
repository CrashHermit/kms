import asyncio
import difflib
import json
import os
from pathlib import Path

os.environ['KMS_MODELS__MODULES__CORRECTOR__BASE_URL'] = (
    'http://localhost:8080/v1'
)
os.environ['KMS_MODELS__MODULES__CORRECTOR__MODEL'] = (
    'openai/unsloth/gemma-4-e4b-it-GGUF'
)
os.environ['KMS_MODELS__MODULES__CORRECTOR__API_KEY'] = 'not-needed'

from kms.construction import corrector
from kms.core import content, llm

ROOT = Path(__file__).parents[1]
GOLD = ROOT / 'data' / 'gold' / 'corrector'
INDEX = json.loads((GOLD / 'index.json').read_text())
BY_ID = {record['id']: record for record in INDEX['records']}

RECORD_ID = 'ode_lebl_diffyqs_p00_v1'


async def main():
    record = BY_ID[RECORD_ID]
    transcription = (GOLD / record['transcription']).read_text()
    corrected = (GOLD / record['corrected']).read_text()
    image_path = ROOT / 'output' / 'gold_pages' / record['page_image']

    module = corrector.Corrector(language_model=llm.module_lm('corrector'))
    page_image = content.load_image(str(image_path))

    output = await module.aforward(
        page_image=page_image, transcription=transcription
    )
    print('=== specialist proposals applied ===')

    edits = list(difflib.ndiff(transcription.splitlines(), output.splitlines()))
    for edit in edits:
        if edit.startswith(('+ ', '- ')):
            print(f'  {edit}')
    print('\n=== diff (output vs gold) ===')
    diff = list(
        difflib.unified_diff(
            output.split('\n'), corrected.split('\n'), lineterm=''
        )
    )
    if not diff:
        print('  (identical to gold)')
    for line in diff:
        print(line)
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
