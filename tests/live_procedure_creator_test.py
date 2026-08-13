import asyncio
import sys

sys.path.insert(0, '.')

from kms.core import llm, models
from kms.graph import db, schema, writer
from kms.ingestion.entity_canonicalizer import rebuild as canonicalize
from kms.ingestion.procedure_creator import create_procedures
from kms.ingestion.triplet_extractor import TripletNode

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
            'Neo4j not configured. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.'
        )
        return

    def _session():
        return db.session()

    async with _session() as session:
        await session.run('MATCH (n) DETACH DELETE n')
    await schema.ensure_schema(_session)

    language_model = llm.pipeline_lm()

    nodes = [models.ASTNode(id=0, type='paragraph', content=CONTENT)]
    triplet_node = TripletNode(language_model=language_model)
    result = await triplet_node.run({'nodes': nodes, 'source': SOURCE})
    triplets = result.get('triplets', [])
    print(f'Extracted {len(triplets)} triplet(s):')
    for t in triplets:
        print(f'  {t.subject} | {t.predicate} | {t.object}')

    await writer.persist_nodes(nodes, SOURCE, session_factory=_session)
    await writer.persist_triplets(triplets, SOURCE, session_factory=_session)
    await writer.persist_chain(nodes, SOURCE, session_factory=_session)

    print('\nRunning canonicalization...')
    canonical_result = await canonicalize(
        threshold=0.8,
        language_model=language_model,
        session_factory=_session,
    )
    print(
        f'  {canonical_result["clusters"]} cluster(s), '
        f'{canonical_result["entities"]} entity/entities'
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
        [statement], SOURCE, session_factory=_session
    )
    print('\nCreated 1 Statement (orphan).')

    print('\nRunning procedure creator...')
    created = await create_procedures(
        _session, language_model=language_model, top_k=5
    )
    print(f'Created {created} procedure(s).')

    if created:
        async with _session() as session:
            steps_result = await session.run(
                'MATCH (s:Statement)-[:HAS_PROCEDURE]->(p:Procedure) '
                '-[:FIRST]->(first:Step) '
                'OPTIONAL MATCH (first)-[:THEN*]->(rest:Step) '
                'RETURN s.uuid AS statement_uuid, '
                'first.text AS first_step, '
                'collect(DISTINCT rest.text) AS more_steps'
            )
            for r in [row async for row in steps_result]:
                print(f'\nStatement: {r["statement_uuid"]}')
                print(f'  Step 0: {r["first_step"]}')
                for i, step in enumerate(r['more_steps'] or [], 1):
                    print(f'  Step {i}: {step}')

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
