"""Live smoke test for the triplet extraction + entity enrichment + embedding
pipeline.

Exercises the real LM and embedding API. Run from the repo root with:
    .venv/bin/python tests/live_triplet_embed_test.py
"""

import asyncio

from kms.core import llm
from kms.ingestion.entity_embedder import embed_descriptions
from kms.ingestion.entity_enricher import EntityEnricher, enrich_entities
from kms.ingestion.triplet_extractor import TripletExtractor, extract_triplets

SAMPLE_FACTS = [
    # Single relation — should yield 1 triplet
    'The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$.',
    # Two relations conjoined — should yield 2 triplets
    'The set $\\mathbb{R}$ is uncountable and has cardinality '
    '$2^{\\aleph_0}$.',
    # Definition with condition — may yield 1 or 2 triplets
    'A function $f$ is continuous at $c$ if '
    '$\\lim_{x\\to c} f(x) = f(c)$.',
    # Simple property
    'The derivative of $\\sin x$ is $\\cos x$.',
    # Instruction-wrapped assertion
    'Prove that every continuous function on $[0,1]$ is bounded.',
    # Compound assertion
    'The graph $G_4$ is NOT a subgraph of $G_1$, even though it looks '
    'like all we did is remove vertex $e$.',
]

# Single-node facts for testing — every fact anchors at node 0.
from kms.core.models import ASTNode

NODE_ID = 0
FAKE_NODES = [
    ASTNode(
        id=0,
        type='paragraph',
        content=(
            'The discriminant is b^2-4ac. The set R is uncountable and '
            'has cardinality 2^{aleph_0}. A function f is continuous '
            'at c. The derivative of sin x is cos x. Every continuous '
            'function on [0,1] is bounded. The graph G4 is NOT a '
            'subgraph of G1.'
        ),
    )
]


async def main():
    lm = llm.text_lm()

    # --- Stage 1: triplet extraction ----------------------------------------
    print('=' * 60)
    print('STAGE 1 — Triplet extraction')
    print('=' * 60)

    extractor = TripletExtractor(language_model=lm)
    triplets = []
    for i, fact_text in enumerate(SAMPLE_FACTS):
        t = await extractor.aforward(fact_text)
        for triplet in t:
            triplet.fact_index = i
        triplets.extend(t)

    from kms.core.models import AtomicFact

    facts = [
        AtomicFact(text=text, node_ids=[NODE_ID])
        for text in SAMPLE_FACTS
    ]

    for i, fact in enumerate(facts):
        print(f'\nFact {i}: {fact.text}')
        my_triplets = [
            t for t in triplets if t.fact_index == i
        ]
        if not my_triplets:
            print('  (no triplets)')
        for j, t in enumerate(my_triplets):
            print(f'  Triplet {j}:')
            print(f'    subject:   {t.subject}')
            print(f'    predicate: {t.predicate}')
            print(f'    object:    {t.object}')

    total = len(triplets)
    print(f'\nTotal: {total} triplet(s) from {len(facts)} fact(s)')

    if not triplets:
        print('No triplets — stopping.')
        return

    # --- Stage 2: entity enrichment -----------------------------------------
    print(f'\n{"=" * 60}')
    print('STAGE 2 — Entity + predicate enrichment')
    print('=' * 60)

    enricher = EntityEnricher(language_model=lm)
    entity_descs, predicate_descs = await enrich_entities(
        triplets=triplets,
        facts=facts,
        nodes=FAKE_NODES,
        module=enricher,
    )

    print(f'\nEntity descriptions (node {NODE_ID}):')
    for entry in entity_descs.get(NODE_ID, []):
        print(f'  {entry["name"]}:')
        print(f'    {entry["description"]}')

    print(f'\nPredicate descriptions (node {NODE_ID}):')
    for entry in predicate_descs.get(NODE_ID, []):
        print(f'  {entry["predicate"]}:')
        print(f'    {entry["description"]}')

    # --- Stage 3: embedding -------------------------------------------------
    print(f'\n{"=" * 60}')
    print('STAGE 3 — Embedding')
    print('=' * 60)

    enriched_entities, enriched_predicates = await embed_descriptions(
        entity_descs, predicate_descs
    )

    print(f'\nEnriched entities (node {NODE_ID}):')
    for entry in enriched_entities.get(NODE_ID, []):
        emb = entry.get('embedding')
        dims = len(emb) if emb else 'N/A'
        print(f'  {entry["name"]}:')
        print(f'    description: {entry.get("description", "(none)")}')
        print(f'    embedding:   {dims}-dimensional vector '
              f'({emb[:3]}...)' if emb else f'    embedding:   N/A')

    print(f'\nEnriched predicates (node {NODE_ID}):')
    for entry in enriched_predicates.get(NODE_ID, []):
        emb = entry.get('embedding')
        dims = len(emb) if emb else 'N/A'
        print(f'  {entry["predicate"]}:')
        print(f'    description: {entry.get("description", "(none)")}')
        print(f'    embedding:   {dims}-dimensional vector '
              f'({emb[:3]}...)' if emb else f'    embedding:   N/A')

    print(f'\n{"=" * 60}')
    print('Done.')


if __name__ == '__main__':
    asyncio.run(main())
