"""
Canonical triplet hub builder — groups canonical hub triplets by their
three hubs, builds ``:TripletHub`` + ``:FactHub`` vertices.

Runs after canonicalization.  Reads the canonical hub graph, groups
every ``:Triplet`` that resolves to the same ``(subj_hub, pred_hub,
obj_hub)``, and writes one ``:TripletHub`` per unique assertion with
``:FactHub`` carrying the assembled canonical text and its embedding.

Design commitments:

* NO LLM.  The assertion text is a deterministic template built from
  the three hub display names — no synthesis needed because the hubs
  already carry canonical names.

* IDEMPOTENT.  ``uuid5(source, subj_hub, pred_hub, obj_hub)`` means
  re-running produces the same hubs.  New ``:Triplet`` nodes add
  ``:SUPPORTED_BY`` edges to existing ``:TripletHub`` nodes.

* ONE PASS.  All groups are processed in a single batch.
"""

import logging

from kms.core import embeddings
from kms.graph.triplet_hubs import triplet_hub_uuid

logger = logging.getLogger(__name__)


async def build_triplet_hubs(
    source: str,
    session_factory,
) -> list[dict]:
    """Build canonical TripletHubs + FactHubs from the hub graph.

    Args:
        source: The stable book identity.
        session_factory: Neo4j session factory.

    Returns:
        One dict per group:
        ``{triplet_hub_uuid, subj_hub, pred_hub, obj_hub,
          triplet_uuids, fact_text, fact_embedding}``.
    """
    from kms.graph import queries

    # --- Read canonical hub triplets -----------------------------------
    print('Reading canonical hub triplets...')
    hub_triplets = await queries.canonical_hub_triplets(
        session_factory, source=source
    )
    print(f'  {len(hub_triplets)} canonical triplet(s) at hub level')

    if not hub_triplets:
        return []

    # --- Group by (subj_hub, pred_hub, obj_hub) ------------------------
    from collections import defaultdict

    groups: dict[
        tuple[str, str, str], list[dict]
    ] = defaultdict(list)
    for ht in hub_triplets:
        key = (ht['subj_hub'], ht['pred_hub'], ht['obj_hub'])
        groups[key].append(ht)

    print(f'  {len(groups)} unique canonical assertion(s) '
          f'({len(hub_triplets)} total triplet(s))')

    # --- Build group dicts ---------------------------------------------
    result: list[dict] = []
    texts_to_embed: list[str] = []
    embed_indices: list[int] = []

    for i, ((subj, pred, obj), triplets) in enumerate(groups.items()):
        first = triplets[0]
        subj_name = first['subj_name']
        pred_name = first['pred_name']
        obj_name = first['obj_name']

        # Assemble canonical assertion text
        fact_text = f'{subj_name} {pred_name} {obj_name}'

        result.append({
            'triplet_hub_uuid': triplet_hub_uuid(
                source, subj, pred, obj
            ),
            'subj_hub': subj,
            'pred_hub': pred,
            'obj_hub': obj,
            'triplet_uuids': [t['triplet_uuid'] for t in triplets],
            'fact_text': fact_text,
        })
        texts_to_embed.append(fact_text)
        embed_indices.append(i)

    # --- Embed assertion texts -----------------------------------------
    if embeddings.is_configured():
        embedder = embeddings.embedder()
        vectors = await embedder.embed(texts_to_embed)
        for idx, vector in zip(embed_indices, vectors, strict=True):
            result[idx]['fact_embedding'] = vector
        print(f'  {len(vectors)} assertion text(s) embedded')
    else:
        print('  Embedding API not configured — skipping embeddings')

    return result
