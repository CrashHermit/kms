import asyncio
import sys

sys.path.insert(0, '.')

from kms.construction import (
    entity_enrichment,
    local_entity_hubs,
    predicate_enrichment,
    triplet_extractor,
)
from kms.core import llm, models, semantic
from kms.graph import db, schema, writer

SOURCE = 'graph_theory_test'

CONTENT = """
We say that $G' = (V', E')$ is a subgraph of $G = (V, E)$, and write
$G' \\subseteq G$, provided $V' \\subseteq V$ and $E' \\subseteq E$.

We say that $G' = (V', E')$ is an induced subgraph of $G = (V, E)$
provided $V' \\subseteq V$ and every edge in $E$ whose vertices are
still in $V'$ is also an edge in $E'$.

Notice that every induced subgraph is also an ordinary subgraph, but
not conversely.

Prove that every induced subgraph is also a subgraph.
"""


async def main():
    if not db.is_configured():
        print(
            'Neo4j not configured. Set KMS_DATABASE__URI, '
            'KMS_DATABASE__USERNAME, KMS_DATABASE__PASSWORD.'
        )
        return

    def _session():
        return db.session()

    async with _session() as session:
        await session.run('MATCH (n) DETACH DELETE n')
    await schema.ensure_schema(_session)

    language_model = llm.module_lm('procedure_enrichment')

    nodes = [models.SourceNode(id=0, type='paragraph', content=CONTENT)]
    triplet_node = triplet_extractor.TripletNode(
        fact_module=triplet_extractor._FactExtractor(language_model),
        triplet_module=triplet_extractor._TripletDecomposer(language_model),
    )
    result = await triplet_node.run({'nodes': nodes, 'source': SOURCE})
    triplets = result.get('triplets', [])
    print(f'Extracted {len(triplets)} triplet(s):')
    for t in triplets:
        print(f'  {t.subject} | {t.predicate} | {t.object}')

    await writer.persist_nodes(nodes, SOURCE, session_factory=_session)

    enricher = entity_enrichment.EntityEnricher(language_model=language_model)
    entity_descriptions = await entity_enrichment.enrich(
        nodes, triplets, enricher
    )
    predicate_descriptions = await predicate_enrichment.enrich(
        nodes,
        triplets,
        predicate_enrichment.PredicateEnricher(language_model=language_model),
    )
    entity_embeddings = await semantic.embed_descriptions(entity_descriptions)
    predicate_embeddings = await semantic.embed_descriptions(
        predicate_descriptions
    )

    await writer.persist_assertions(
        triplets,
        SOURCE,
        nodes,
        session_factory=_session,
        entity_descriptions=entity_descriptions,
        predicate_descriptions=predicate_descriptions,
        entity_embeddings=entity_embeddings,
        predicate_embeddings=predicate_embeddings,
    )
    await writer.persist_chain(nodes, SOURCE, session_factory=_session)

    print('\nRunning hub building...')
    hub_result = await local_entity_hubs.rebuild_source(
        SOURCE,
        language_model=language_model,
        synthesizer=local_entity_hubs.EntityHubSynthesizer(
            language_model=language_model
        ),
        session_factory=_session,
    )
    print(
        f'  {hub_result["clusters"]} cluster(s), '
        f'{hub_result["records"]} record(s)'
    )

    async with _session() as session:
        hubs = await session.run(
            'MATCH (h:EntityHub) '
            'RETURN h.canonical_name AS name, h.description AS desc'
        )
        hub_records = [r async for r in hubs]
        print(f'\n{len(hub_records)} EntityHub(s):')
        for r in hub_records:
            print(f'  {r["name"]}: {r["desc"][:120]}')

    statement = models.Statement(block=[0], members=[0])
    await writer.persist_statements(
        [statement], nodes, SOURCE, session_factory=_session
    )
    print('\nCreated 1 Statement (orphan).')

    print(
        '\nProcedure content is now produced by ProcedureEnricher in the workflow.'
    )

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
