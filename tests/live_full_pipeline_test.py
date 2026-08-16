import asyncio
from pathlib import Path

from kms.construction import canonicalizer
from kms.construction.canonicalizer import rebuild
from kms.construction.triplet_extractor import (
    TripletNode,
    _FactExtractor,
    _TripletDecomposer,
)
from kms.core import embeddings, llm, models
from kms.graph import db, schema, writer

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

    lm = llm.module_lm('triplet_extractor')
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

        triplet_node = TripletNode(
            fact_module=_FactExtractor(lm),
            triplet_module=_TripletDecomposer(lm),
        )
        result = await triplet_node.run({'nodes': [node], 'source': SOURCE})
        triplets = result.get('triplets', [])
        all_triplets.extend(triplets)
        print(f'Page {page_name}: {len(triplets)} triplets')

    print(f'\nTotal: {len(all_triplets)} triplets from {len(PAGES)} pages')

    await writer.persist_nodes(all_nodes, SOURCE, session_factory=_session)

    enricher = canonicalizer.ComponentEnricher(language_model=lm)
    (
        entity_descriptions,
        predicate_descriptions,
    ) = await canonicalizer.enrich_components(all_nodes, all_triplets, enricher)

    (
        entity_embeddings,
        predicate_embeddings,
    ) = await canonicalizer.embed_components(
        entity_descriptions,
        predicate_descriptions,
        embeddings.embedder(),
    )

    await writer.persist_assertions(
        all_triplets,
        SOURCE,
        session_factory=_session,
        entity_descriptions=entity_descriptions,
        predicate_descriptions=predicate_descriptions,
        entity_embeddings=entity_embeddings,
        predicate_embeddings=predicate_embeddings,
    )
    await writer.persist_chain(all_nodes, SOURCE, session_factory=_session)

    print('\n' + '=' * 60)
    print('CANONICALIZER')
    print('=' * 60)

    result = await rebuild(
        'entity',
        language_model=lm,
        session_factory=_session,
        source=SOURCE,
    )
    print(f'\nRecords: {result["records"]}')
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
