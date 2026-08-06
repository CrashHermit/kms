"""Live test: entity and predicate canonicalization on the graph-theory
page embeddings.

Runs clustering + definition synthesis on the previously-computed entity
and predicate embeddings from the combinatorics page. Requires the
embedding API key to be configured for definition-embedding.

Run from the repo root with:
    .venv/bin/python tests/live_canonicalize_test.py
"""

import asyncio

from kms.core import llm
from kms.ingestion.canonicalizer import (
    canonicalize_entities,
    canonicalize_predicates,
)

# Entity spokes from the previous pipeline run (combinatorics graph-theory
# page, node 0). Each has uuid, name, description, and embedding from the
# entity_embedder stage.  Real embeddings omitted here for brevity — the
# test re-runs the full pipeline to get fresh embeddings.

# Predicate spokes from the same run.


async def main():
    print('=' * 60)
    print('Running full semantic pipeline to get spokes...')
    print('=' * 60)

    # Re-run the semantic pipeline to get fresh embeddings
    from kms.core import models
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

    page_content = """\
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

    nodes = [
        models.ASTNode(
            id=0, type='paragraph', content=page_content, segment_index=0
        )
    ]

    language_model = llm.text_lm()

    # --- Run pipeline to get embeddings ---
    af_module = AtomicFactExtractor(language_model=language_model)
    af_node = AtomicFactNode(module=af_module)
    result = await af_node.run({'nodes': nodes})
    facts = result.get('atomic_facts', [])

    t_module = TripletExtractor(language_model=language_model)
    triplets = await extract_triplets(facts, module=t_module)

    enricher = EntityEnricher(language_model=language_model)
    entity_descs, predicate_descs = await enrich_entities(
        triplets=triplets, facts=facts, nodes=nodes, module=enricher
    )

    enriched_entities, enriched_predicates = await embed_descriptions(
        entity_descs, predicate_descs
    )

    # --- Build spoke lists ---
    entity_spokes: list[dict] = []
    from kms.graph.entities import entity_uuid
    from kms.graph.facts import fact_uuid as fact_uuid_fn
    from kms.graph.triplets import triplet_uuid as triplet_uuid_fn

    source = 'combinatorics_levin'
    for node_id, entries in enriched_entities.items():
        for entry in entries:
            # Build a deterministic spoke uuid matching the real graph
            fact_index = 0
            fact_uid = fact_uuid_fn(source, [node_id], fact_index)
            triplet_uid = triplet_uuid_fn(
                source, node_id, fact_uid, entry['name'], '', ''
            )
            spoke_uuid = entity_uuid(
                source, node_id, triplet_uid, 'subject'
            )
            entity_spokes.append({
                'uuid': spoke_uuid,
                'name': entry['name'],
                'description': entry.get('description'),
                'embedding': entry.get('embedding'),
            })

    predicate_spokes: list[dict] = []
    from kms.graph.predicates import predicate_uuid as pred_uuid_fn

    for node_id, entries in enriched_predicates.items():
        for entry in entries:
            fact_uid = fact_uuid_fn(source, [node_id], 0)
            spoke_uuid = pred_uuid_fn(
                source, node_id, fact_uid, '', entry['predicate'], ''
            )
            predicate_spokes.append({
                'uuid': spoke_uuid,
                'predicate': entry['predicate'],
                'description': entry.get('description'),
                'embedding': entry.get('embedding'),
            })

    total_entities = len(entity_spokes)
    total_predicates = len(predicate_spokes)
    print(f'{total_entities} entity spoke(s), '
          f'{total_predicates} predicate spoke(s)')

    # --- Canonicalize entities ---
    threshold = 0.85
    print(f'\n{"=" * 60}')
    print(f'Entity canonicalization (threshold={threshold})')
    print('=' * 60)

    entity_clusters, entity_defs = await canonicalize_entities(
        entity_spokes, threshold, source, language_model
    )

    print(f'\n{len(entity_clusters)} entity cluster(s):')
    for i, cluster in enumerate(entity_clusters):
        names = sorted(set(s['name'] for s in cluster))
        definition = entity_defs[i]
        print(f'\n  Cluster {i} ({len(cluster)} spoke(s)):')
        print(f'    Display name: {definition["display_name"]}')
        print(f'    Members: {names}')
        print(f'    Definition: {definition["definition_text"]}')
        emb = definition.get('definition_embedding')
        if emb:
            print(f'    Embedding: {len(emb)}-d vector '
                  f'({emb[:3]}...)')

    # --- Canonicalize predicates ---
    print(f'\n{"=" * 60}')
    print(f'Predicate canonicalization (threshold={threshold})')
    print('=' * 60)

    predicate_clusters, predicate_defs = await canonicalize_predicates(
        predicate_spokes, threshold, source, language_model
    )

    print(f'\n{len(predicate_clusters)} predicate cluster(s):')
    for i, cluster in enumerate(predicate_clusters):
        names = sorted(set(s['predicate'] for s in cluster))
        definition = predicate_defs[i]
        print(f'\n  Cluster {i} ({len(cluster)} spoke(s)):')
        print(f'    Display name: {definition["display_name"]}')
        print(f'    Members: {names}')
        print(f'    Definition: {definition["definition_text"]}')
        emb = definition.get('definition_embedding')
        if emb:
            print(f'    Embedding: {len(emb)}-d vector '
                  f'({emb[:3]}...)')

    print(f'\n{"=" * 60}')
    print('Done.')


if __name__ == '__main__':
    asyncio.run(main())
