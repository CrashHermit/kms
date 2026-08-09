import logging

from kms.core import embeddings
from kms.graph import triplet_hubs

logger = logging.getLogger(__name__)


async def build_triplet_hubs(
    source: str,
    session_factory,
) -> list[dict]:
    from kms.graph import queries
    print('Reading canonical hub triplets...')
    hub_triplets = await queries.canonical_hub_triplets(
        session_factory, source=source
    )
    print(f'  {len(hub_triplets)} canonical triplet(s) at hub level')

    if not hub_triplets:
        return []
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for hub_triplet in hub_triplets:
        key = (
            hub_triplet['subj_hub'],
            hub_triplet['pred_hub'],
            hub_triplet['obj_hub'],
        )
        groups[key].append(hub_triplet)

    print(
        f'  {len(groups)} unique canonical assertion(s) '
        f'({len(hub_triplets)} total triplet(s))'
    )
    result: list[dict] = []
    texts_to_embed: list[str] = []
    embed_indices: list[int] = []

    for i, ((subj, pred, obj), triplets) in enumerate(groups.items()):
        first = triplets[0]
        subj_name = first['subj_name']
        pred_name = first['pred_name']
        obj_name = first['obj_name']
        fact_text = f'{subj_name} {pred_name} {obj_name}'

        result.append(
            {
                'triplet_hub_uuid': triplet_hubs.triplet_hub_uuid(
                    source, subj, pred, obj
                ),
                'subj_hub': subj,
                'pred_hub': pred,
                'obj_hub': obj,
                'triplet_uuids': [t['triplet_uuid'] for t in triplets],
                'fact_text': fact_text,
            }
        )
        texts_to_embed.append(fact_text)
        embed_indices.append(i)
    if embeddings.is_configured():
        embedder = embeddings.embedder()
        vectors = await embedder.embed(texts_to_embed)
        for index, vector in zip(embed_indices, vectors, strict=True):
            result[index]['fact_embedding'] = vector
        print(f'  {len(vectors)} assertion text(s) embedded')
    else:
        print('  Embedding API not configured — skipping embeddings')

    return result

