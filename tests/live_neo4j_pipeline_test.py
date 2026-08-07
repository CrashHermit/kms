"""Live test: semantic pipeline on cached content through to Neo4j.

Uses the combinatorics graph-theory page content from
``live_semantic_pipeline_test.py`` — already in the DSPy cache, so no
new LLM calls. Runs atomic-fact extraction → triplet extraction →
entity enrichment → embedding, then persists Entities, Predicates,
Triplets, and Facts to Neo4j and verifies.

Requires NEO4J_URI/USERNAME/PASSWORD and OPENROUTER_API_KEY (for
embeddings) configured in .env.

Run from the repo root with:
    .venv/bin/python tests/live_neo4j_pipeline_test.py
"""

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# The same content as live_semantic_pipeline_test.py — the corrected
# transcription of the combinatorics graph-theory page.
PAGE_CONTENT = """\
Here both $G_2$ and $G_3$ are subgraphs of $G_1$. But only $G_2$ is an \
*induced* subgraph. Every edge in $G_1$ that connects vertices in $G_2$ is \
also an edge in $G_2$. In $G_3$, the edge $\\{a, b\\}$ is in $E_1$ but not \
$E_3$, even though vertices $a$ and $b$ are in $V_3$.

The graph $G_4$ is NOT a subgraph of $G_1$, even though it looks like all we \
did is remove vertex $e$. The reason is that in $E_4$ we have the edge \
$\\{c, f\\}$, but this is not an element of $E_1$, so we don't have the \
required $E_4 \\subseteq E_1$.

Back to some basic graph theory definitions. Notice that all the graphs we \
have drawn above have the property that no pair of vertices is connected \
more than once, and no vertex is connected to itself. Graphs like these are \
sometimes called **simple**, although we will just call them *graphs*.

The graphs above are also **connected**: you can get from any vertex to any \
other vertex by following some path of edges. A graph that is not connected \
can be thought of as two separate graphs drawn close together."""

SOURCE = 'combinatorics_graph_theory_page'

# Single-node anchor: the whole page as one paragraph node.
FAKE_NODES = [
    type(
        'ASTNode',
        (),
        {'id': 0, 'type': 'paragraph', 'content': PAGE_CONTENT,
         'segment_index': 0},
    )()
]


async def main() -> None:
    from kms.core import llm, models
    from kms.graph import db, schema, writer
    from kms.ingestion.atomic_fact_extractor import (
        AtomicFactExtractor,
        AtomicFactNode,
    )
    from kms.ingestion.entity_embedder import embed_descriptions
    from kms.ingestion.entity_enricher import (
        EntityEnricher,
        enrich_entities,
    )
    from kms.ingestion.triplet_extractor import (
        TripletExtractor,
        extract_triplets,
    )

    lm = llm.text_lm()

    # -- Ensure schema exists ------------------------------------------------
    if db.is_configured():

        def _session_factory():
            return db.session()

        await schema.ensure_schema(_session_factory)

    # ======================================================================
    # STAGE 1 — Atomic facts
    # ======================================================================
    print('=' * 60)
    print('STAGE 1 — Atomic facts')
    print('=' * 60)

    nodes = [
        models.ASTNode(
            id=0, type='paragraph', content=PAGE_CONTENT, segment_index=0
        )
    ]
    af_module = AtomicFactExtractor(language_model=lm)
    af_node = AtomicFactNode(module=af_module)
    result = await af_node.run({'nodes': nodes})
    facts = result.get('atomic_facts', [])

    print(f'\n{len(facts)} atomic fact(s):')
    for i, fact in enumerate(facts):
        print(f'  [{i}] {fact.text}')

    if not facts:
        print('No facts — stopping.')
        return

    # ======================================================================
    # STAGE 2 — Triplet extraction
    # ======================================================================
    print(f'\n{"=" * 60}')
    print('STAGE 2 — Triplet extraction')
    print('=' * 60)

    t_module = TripletExtractor(language_model=lm)
    triplets = await extract_triplets(facts, module=t_module)

    for i, fact in enumerate(facts):
        my_triplets = [t for t in triplets if t.fact_index == i]
        if not my_triplets:
            print(f'\n  Fact [{i}]: (no triplets)')
            continue
        print(f'\n  Fact [{i}]: {fact.text}')
        for j, t in enumerate(my_triplets):
            print(f'    Triplet {j}: ({t.subject}) --[{t.predicate}]--> '
                  f'({t.object})')

    print(f'\nTotal: {len(triplets)} triplet(s) from {len(facts)} fact(s)')

    if not triplets:
        print('No triplets — stopping.')
        return

    # ======================================================================
    # STAGE 3 — Entity + predicate enrichment
    # ======================================================================
    print(f'\n{"=" * 60}')
    print('STAGE 3 — Entity + predicate enrichment')
    print('=' * 60)

    e_module = EntityEnricher(language_model=lm)
    entity_descs, predicate_descs = await enrich_entities(
        triplets=triplets, facts=facts, nodes=nodes, module=e_module
    )

    n_entity = sum(len(v) for v in entity_descs.values())
    n_pred = sum(len(v) for v in predicate_descs.values())
    print(f'\n{n_entity} entity description(s), {n_pred} predicate '
          f'description(s)')

    for node_id, entries in entity_descs.items():
        print(f'\n  Entities (node {node_id}):')
        for entry in entries:
            print(f'    {entry["name"]}: {entry["description"]}')

    for node_id, entries in predicate_descs.items():
        print(f'\n  Predicates (node {node_id}):')
        for entry in entries:
            print(f'    {entry["predicate"]}: {entry["description"]}')

    # ======================================================================
    # STAGE 4 — Embedding
    # ======================================================================
    print(f'\n{"=" * 60}')
    print('STAGE 4 — Embedding')
    print('=' * 60)

    enriched_entities, enriched_predicates = await embed_descriptions(
        entity_descs, predicate_descs
    )

    embedded_ent = sum(
        1 for entries in enriched_entities.values()
        for e in entries if e.get('embedding')
    )
    embedded_pred = sum(
        1 for entries in enriched_predicates.values()
        for e in entries if e.get('embedding')
    )
    print(f'{embedded_ent}/{n_entity} entity embeddings, '
          f'{embedded_pred}/{n_pred} predicate embeddings')

    # ======================================================================
    # STAGE 5 — Persist to Neo4j
    # ======================================================================
    if not db.is_configured():
        print('\nNeo4j not configured — stopping.')
        return

    print(f'\n{"=" * 60}')
    print('STAGE 5 — Persist to Neo4j')
    print('=' * 60)

    def _sf():
        return db.session()

    # Write nodes first (entities/predicates reference them)
    await writer.persist_nodes(nodes, SOURCE, session_factory=_sf)
    print('  :Source + :Node written')

    # Write the provenance chain
    await writer.persist_chain(nodes, SOURCE, session_factory=_sf)
    print('  :NEXT chain written')

    # Write facts
    await writer.persist_facts(facts, SOURCE, session_factory=_sf)
    print(f'  {len(facts)} :Fact(s) written')

    # Write triplets FIRST — Predicate edges reference Triplet hubs
    await writer.persist_triplets(triplets, facts, SOURCE, session_factory=_sf)
    async with db.session() as s:
        r = await s.run('MATCH (t:Triplet) RETURN count(t) AS cnt')
        trip_cnt = (await r.single())['cnt']
    print(f'  {trip_cnt} :Triplet hubs written')

    # Write entities (needs triplets, facts, enriched descriptions)
    await writer.persist_entities(
        triplets, facts, enriched_entities, SOURCE, session_factory=_sf,
    )
    async with db.session() as s:
        r = await s.run('MATCH (e:Entity) RETURN count(e) AS cnt')
        ent_cnt = (await r.single())['cnt']
    print(f'  {ent_cnt} :Entity vertices written')

    # Write predicates AFTER triplets — HAS_PREDICATE edges need both
    await writer.persist_predicates(
        triplets, facts, SOURCE,
        session_factory=_sf,
        node_predicate_descriptions=enriched_predicates,
    )
    async with db.session() as s:
        r = await s.run('MATCH (p:Predicate) RETURN count(p) AS cnt')
        pred_cnt = (await r.single())['cnt']
    print(f'  {pred_cnt} :Predicate vertices written')

    # ======================================================================
    # STAGE 6 — Verify
    # ======================================================================
    print(f'\n{"=" * 60}')
    print('NEO4J VERIFICATION')
    print('=' * 60)

    async with db.session() as session:

        async def _count(label: str) -> int:
            r = await session.run(
                f'MATCH (n:`{label}`) RETURN count(n) AS cnt'
            )
            return (await r.single())['cnt']

        async def _edge_count(edge: str) -> int:
            r = await session.run(
                f'MATCH ()-[r:`{edge}`]->() RETURN count(r) AS cnt'
            )
            return (await r.single())['cnt']

        labels = ['Source', 'Node', 'Fact', 'Entity', 'Predicate', 'Triplet']
        for label in labels:
            cnt = await _count(label)
            print(f'  :{label:<12} {cnt}')

        print()
        edges = ['NEXT', 'HEAD', 'EVIDENCE_FOR', 'YIELDS',
                 'HAS_SUBJECT', 'HAS_OBJECT', 'HAS_PREDICATE']
        for edge in edges:
            cnt = await _edge_count(edge)
            print(f'  [:{edge:<16}] {cnt}')

        # Spot-check: list all triplets with their subject/predicate/object
        print()
        print('  --- All Triplets ---')
        r = await session.run(
            'MATCH (t:Triplet) '
            'OPTIONAL MATCH (t)-[:HAS_SUBJECT]->(subj:Entity) '
            'OPTIONAL MATCH (t)-[:HAS_OBJECT]->(obj:Entity) '
            'OPTIONAL MATCH (t)-[:HAS_PREDICATE]->(pred:Predicate) '
            'RETURN t.uuid, subj.name AS subject, '
            'pred.predicate AS predicate, obj.name AS object '
            'ORDER BY t.uuid'
        )
        async for rec in r:
            print(f'  ({rec["subject"]}) --[{rec["predicate"]}]--> '
                  f'({rec["object"]})')

        # Check embedding coverage
        r = await session.run(
            'MATCH (e:Entity) WHERE e.embedding IS NOT NULL '
            'RETURN count(e) AS cnt'
        )
        ent_emb = (await r.single())['cnt']
        r = await session.run(
            'MATCH (p:Predicate) WHERE p.embedding IS NOT NULL '
            'RETURN count(p) AS cnt'
        )
        pred_emb = (await r.single())['cnt']
        print()
        print(f'  Entities with embedding:   {ent_emb}/{ent_cnt}')
        print(f'  Predicates with embedding: {pred_emb}/{pred_cnt}')

        # Spot-check a sample embedding dimension
        r = await session.run(
            'MATCH (e:Entity) WHERE e.embedding IS NOT NULL '
            'RETURN e.name, size(e.embedding) AS dims LIMIT 2'
        )
        async for rec in r:
            print(f'  Sample entity "{rec["e.name"]}": '
                  f'{rec["dims"]}-dimensional embedding')

    await db.close_driver()

    print()
    print('=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
