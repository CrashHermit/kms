import asyncio
from collections import Counter
from pathlib import Path

from kms import runtime
from kms.graph import db, schema

PDF = Path('tests/fixtures/books/combinatorics_levin.pdf')
OUTPUT = 'output/live_full_pipeline'
PAGES = [0, 1, 2]
SOURCE = 'combinatorics_levin_ch2'


async def main():
    if not PDF.is_file():
        raise FileNotFoundError(f'PDF fixture does not exist: {PDF}')
    if not db.is_configured():
        print('Neo4j not configured.')
        return

    print('Clearing Neo4j...')
    async with db.session() as session:
        result = await session.run('MATCH (n) DETACH DELETE n')
        await result.consume()
    await schema.ensure_schema(lambda: db.session())

    print(f'Ingesting {PDF} pages {PAGES}...')
    result = await runtime.ingest(
        PDF,
        output_dir=OUTPUT,
        pages=PAGES,
        source=SOURCE,
    )

    nodes = result.get('nodes') or []
    print(f'\nNodes: {len(nodes)}')
    print('Node types:', dict(Counter(str(node.type) for node in nodes)))

    triplets = result.get('triplets') or []
    print(f'Triplets: {len(triplets)}')
    for triplet in triplets:
        print(f'  {triplet.subject} | {triplet.predicate} | {triplet.object}')

    print('\nPipeline results:')
    for name in (
        'statements_enriched',
        'procedures_enriched',
        'statement_hubs_created',
        'procedure_hubs_created',
        'entity_assigned',
        'predicate_assigned',
        'procedures_created',
    ):
        if name in result:
            print(f'  {name}: {result[name]}')

    async with db.session() as session:
        hubs = await session.run(
            'MATCH (h:EntityHub) '
            'RETURN h.canonical_name AS name ORDER BY name'
        )
        records = [record async for record in hubs]
        print(f'\nEntity hubs: {len(records)}')
        for record in records:
            print(f'  {record["name"]}')

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())