import asyncio
import os
from collections import Counter

os.environ['KMS_MODELS__MODULES__FORMATTER__BASE_URL'] = (
    'http://127.0.0.1:8080/v1'
)
os.environ['KMS_MODELS__MODULES__FORMATTER__MODEL'] = 'openai/gemma-4-e4b-it'
os.environ['KMS_MODELS__MODULES__FORMATTER__API_KEY'] = 'not-needed'
os.environ['KMS_MODELS__MODULES__CORRECTOR__BASE_URL'] = (
    'http://127.0.0.1:8080/v1'
)
os.environ['KMS_MODELS__MODULES__CORRECTOR__MODEL'] = 'openai/qwen3-vl-4b'
os.environ['KMS_MODELS__MODULES__CORRECTOR__API_KEY'] = 'not-needed'
os.environ['KMS_MODELS__MODULES__PROCEDURE_CREATOR__BASE_URL'] = (
    'http://127.0.0.1:8080/v1'
)
os.environ['KMS_MODELS__MODULES__PROCEDURE_CREATOR__MODEL'] = (
    'openai/qwen3.5-9b'
)
os.environ['KMS_MODELS__MODULES__PROCEDURE_CREATOR__API_KEY'] = 'not-needed'
os.environ['KMS_SERVING__MANAGE'] = '1'
os.environ['KMS_SERVING__MODULE_MODELS__FORMATTER'] = 'gemma-4-e4b-it'

from kms import runtime
from kms.graph import db, schema

PDF = 'tests/fixtures/books/logic_hammack_truthtables.pdf'
SOURCE = 'logic_hammack_truthtables'


async def main():
    async with db.session() as session:
        await session.run('MATCH (n) DETACH DELETE n')
    await schema.ensure_schema(lambda: db.session())

    result = await runtime.ingest(
        PDF,
        output_dir='output/live_logic_hammack_e4b',
        source=SOURCE,
        pages=[0, 1, 2, 3],
    )

    nodes = result.get('nodes') or []
    print(f'\n=== nodes: {len(nodes)} ===')
    print('by type:', dict(Counter(node.type for node in nodes)))

    triplets = result.get('triplets') or []
    print(f'\n=== triplets: {len(triplets)} ===')
    for t in triplets:
        print(f'  {t.subject} | {t.predicate} | {t.object}')

    statements = result.get('statements') or []
    print(f'\n=== statements: {len(statements)} ===')

    print('\n=== canonicalization ===')
    print(f'  entity assigned: {result.get("entity_assigned")}')
    print(f'  predicate assigned: {result.get("predicate_assigned")}')

    async with db.session() as session:
        hubs = await session.run(
            'MATCH (h:EntityHub) RETURN h.canonical_name AS name ORDER BY name'
        )
        records = [r async for r in hubs]
        print(f'\n=== {len(records)} EntityHub(s) ===')
        for r in records:
            print(f'  {r["name"]}')
        phubs = await session.run(
            'MATCH (h:PredicateHub) '
            'RETURN h.canonical_name AS name ORDER BY name'
        )
        precords = [r async for r in phubs]
        print(f'\n=== {len(precords)} PredicateHub(s) ===')
        for r in precords:
            print(f'  {r["name"]}')

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
