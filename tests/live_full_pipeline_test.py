import asyncio
from pathlib import Path

from kms.core import llm, models
from kms.graph import db, schema, writer
from kms.ingestion.entity_canonicalizer import rebuild
from kms.ingestion.triplet_extractor import TripletNode

PAGES = [
    ('combinatorics_levin_p00', 'transcription'),
    ('combinatorics_levin_p01', 'transcription'),
    ('combinatorics_levin_p02', 'transcription'),
]

SOURCE = 'combinatorics_levin_ch2'
GOLD = Path('data/gold/corrector/real')


async def main():
    if not db.is_configured():
        print('Neo4j not configured.')
        return

    def _session():
        return db.session()

    async with _session() as session:
        await session.run('MATCH (n) DETACH DELETE n')
    await schema.ensure_schema(_session)

    lm = llm.pipeline_lm()
    all_triplets = []
    all_nodes = []
    node_id = 0

    for page_name, kind in PAGES:
        path = GOLD / page_name / f'{kind}.md'
        content = path.read_text()

        first_header = -1
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('#'):
                first_header = i
                break
        if first_header > 5:
            content = '\n'.join(lines[first_header:])

        node = models.ASTNode(id=node_id, type='paragraph', content=content)
        all_nodes.append(node)
        node_id += 1

        triplet_node = TripletNode(language_model=lm)
        result = await triplet_node.run({'nodes': [node], 'source': SOURCE})
        triplets = result.get('triplets', [])
        all_triplets.extend(triplets)
        print(f'Page {page_name}: {len(triplets)} triplets')

    print(f'\nTotal: {len(all_triplets)} triplets from {len(PAGES)} pages')

    await writer.persist_nodes(all_nodes, SOURCE, session_factory=_session)
    await writer.persist_triplets(
        all_triplets, SOURCE, session_factory=_session
    )
    await writer.persist_chain(all_nodes, SOURCE, session_factory=_session)

    print('\n' + '=' * 60)
    print('CANONICALIZER')
    print('=' * 60)

    result = await rebuild(
        threshold=0.8, language_model=lm, session_factory=_session
    )
    print(f'\nEntities: {result["entities"]}')
    print(f'Clusters: {result["clusters"]}')

    async with _session() as session:
        hubs = await session.run(
            'MATCH (h:EntityHub) '
            'RETURN h.canonical_name AS name, h.description AS description '
            'ORDER BY name'
        )
        records = [r async for r in hubs]
        print(f'\n{len(records)} EntityHub(s):')
        for r in records:
            print(f'  {r["name"]}')
            print(f'    {r["description"]}')
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
