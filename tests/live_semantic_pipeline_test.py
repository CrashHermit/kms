"""Live test: full semantic pipeline on real textbook content.

Runs atomic-fact extraction → triplet extraction → entity enrichment →
embedding on the combinatorics graph-theory page from the gold data.

Run from the repo root with:
    .venv/bin/python tests/live_semantic_pipeline_test.py
"""

import asyncio

from kms.core import llm, models
from kms.ingestion.atomic_fact_extractor import (
    AtomicFactExtractor,
    AtomicFactNode,
)
from kms.ingestion.entity_embedder import embed_descriptions
from kms.ingestion.entity_enricher import EntityEnricher, enrich_entities
from kms.ingestion.triplet_extractor import TripletExtractor, extract_triplets


async def main():
    # Real content from the combinatorics graph-theory page (corrected
    # transcription). The figures are placeholders; the semantic passes
    # ignore them.
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

    # Create a single node with this content.
    nodes = [
        models.ASTNode(
            id=0, type='paragraph', content=page_content, segment_index=0
        )
    ]

    lm = llm.text_lm()

    # --- Stage 1: Atomic facts ----------------------------------------------
    print('=' * 60)
    print('STAGE 1 — Atomic facts')
    print('=' * 60)

    af_module = AtomicFactExtractor(language_model=lm)
    af_node = AtomicFactNode(module=af_module)
    state = {'nodes': nodes}
    result = await af_node.run(state)
    facts = result.get('atomic_facts', [])

    print(f'\n{len(facts)} atomic fact(s):')
    for i, fact in enumerate(facts):
        print(f'  [{i}] {fact.text}')

    if not facts:
        print('No facts — stopping.')
        return

    # --- Stage 2: Triplet extraction ----------------------------------------
    print(f'\n{"=" * 60}')
    print('STAGE 2 — Triplet extraction')
    print('=' * 60)

    t_module = TripletExtractor(language_model=lm)
    triplets = await extract_triplets(facts, module=t_module)

    # Group triplets by fact for display.
    for i, fact in enumerate(facts):
        my_triplets = [t for t in triplets if t.fact_index == i]
        print(f'\n  Fact [{i}]: {fact.text}')
        if not my_triplets:
            print('    (no triplets)')
        for j, t in enumerate(my_triplets):
            print(f'    Triplet {j}:')
            print(f'      subj: {t.subject}')
            print(f'      pred: {t.predicate}')
            print(f'      obj:  {t.object}')

    total = len(triplets)
    print(f'\nTotal: {total} triplet(s) from {len(facts)} fact(s)')

    if not triplets:
        print('No triplets — stopping.')
        return

    # --- Stage 3: Entity + predicate enrichment ----------------------------
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
            print(f'    {entry["name"]}:')
            print(f'      {entry["description"]}')

    for node_id, entries in predicate_descs.items():
        print(f'\n  Predicates (node {node_id}):')
        for entry in entries:
            print(f'    {entry["predicate"]}:')
            print(f'      {entry["description"]}')

    # --- Stage 4: Embedding -------------------------------------------------
    print(f'\n{"=" * 60}')
    print('STAGE 4 — Embedding')
    print('=' * 60)

    enriched_entities, enriched_predicates = await embed_descriptions(
        entity_descs, predicate_descs
    )

    embedded_ent = sum(
        1
        for entries in enriched_entities.values()
        for e in entries
        if e.get('embedding')
    )
    embedded_pred = sum(
        1
        for entries in enriched_predicates.values()
        for e in entries
        if e.get('embedding')
    )
    print(f'{embedded_ent}/{n_entity} entity embeddings, '
          f'{embedded_pred}/{n_pred} predicate embeddings')

    # Show a few sample embeddings.
    if embedded_ent:
        entry = next(
            e
            for entries in enriched_entities.values()
            for e in entries
            if e.get('embedding')
        )
        emb = entry['embedding']
        print(f'\n  Sample entity "{entry["name"]}":')
        print(f'    {len(emb)}-d vector ({emb[:3]}...)')

    if embedded_pred:
        entry = next(
            e
            for entries in enriched_predicates.values()
            for e in entries
            if e.get('embedding')
        )
        emb = entry['embedding']
        print(f'\n  Sample predicate "{entry["predicate"]}":')
        print(f'    {len(emb)}-d vector ({emb[:3]}...)')

    print(f'\n{"=" * 60}')
    print('Done.')


if __name__ == '__main__':
    asyncio.run(main())
