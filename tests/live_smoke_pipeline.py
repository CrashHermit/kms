import asyncio
import os
from collections import Counter

os.environ['KMS_RECORDING__ENABLED'] = 'true'

from kms import runtime
from kms.core import loading
from kms.graph import db, schema

PDF = 'tests/fixtures/books/calc3_gradients_exercises.pdf'
SOURCE = 'calc3_gradients_exercises'


async def main():
    if not db.is_configured():
        print('Neo4j not configured.')
        return

    async with db.session() as session:
        result = await session.run('MATCH (n) DETACH DELETE n')
        await result.consume()
    await schema.ensure_schema(lambda: db.session())

    result = await runtime.ingest(
        PDF,
        output_dir='output/live_smoke',
        source=SOURCE,
        pages=[0, 1],
    )

    nodes = result.get('nodes') or []
    print(f'\n=== nodes: {len(nodes)} ===')
    print('by type:', dict(Counter(node.type for node in nodes)))

    triplets = result.get('triplets') or []
    print(f'\n=== triplets: {len(triplets)} ===')
    for triplet in triplets:
        print(f'  {triplet.subject} | {triplet.predicate} | {triplet.object}')

    print('\n=== canonicalization ===')
    print(f'  entity assigned: {result.get("entity_assigned")}')
    print(f'  predicate assigned: {result.get("predicate_assigned")}')
    print(f'  procedures created: {result.get("procedures_created")}')

    datasets = loading.load_datasets('output/live_smoke/examples')
    print(f'\n=== recorded datasets: {len(datasets)} ===')
    for dataset in datasets:
        print(
            f'  {dataset.stage}: {len(dataset.examples)} example(s), '
            f'signature={dataset.signature.__name__}'
        )

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
