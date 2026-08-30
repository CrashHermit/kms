import asyncio
from collections import Counter
from pathlib import Path
from time import perf_counter

from kms import runtime
from kms.graph import db, schema

PDF = Path('books/01_CEN4020.pdf')
SOURCE = 'cen4020'
OUTPUT = 'output/live_cen4020_reranker'


async def main() -> None:
    if not PDF.is_file():
        raise FileNotFoundError(PDF)
    if not db.is_configured():
        print('Neo4j not configured.')
        return

    async with db.session() as session:
        result = await session.run('MATCH (n) DETACH DELETE n')
        await result.consume()
    await schema.ensure_schema(lambda: db.session())

    started = perf_counter()
    print(f'Ingesting {PDF} ({PDF.stat().st_size} bytes)...', flush=True)
    result = await runtime.ingest(PDF, output_dir=OUTPUT, source=SOURCE)
    elapsed = perf_counter() - started

    nodes = result.get('nodes') or []
    print(f'Elapsed: {elapsed:.2f}s', flush=True)
    print(f'Nodes: {len(nodes)}', flush=True)
    print('Node types:', dict(Counter(str(node.type) for node in nodes)), flush=True)
    for key in (
        'triplets',
        'statements_enriched',
        'procedures_enriched',
        'statement_hubs_created',
        'procedure_hubs_created',
        'entity_assigned',
        'predicate_assigned',
        'procedures_created',
    ):
        if key in result:
            value = result[key]
            print(f'{key}: {len(value) if isinstance(value, list) else value}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
