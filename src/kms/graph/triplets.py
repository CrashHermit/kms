from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

TRIPLET_LABEL = 'Triplet'


def triplet_uuid(
    source: str,
    node_id: int,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#triplet#{node_id}#{subject}#{predicate}#{object}',
    ).hex


def triplet_properties(
    triplet: models.Triplet,
    source: str,
    node_id: int,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': triplet_uuid(
            source,
            node_id,
            triplet.subject,
            triplet.predicate,
            triplet.object,
        ),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'subject': triplet.subject,
        'predicate': triplet.predicate,
        'object': triplet.object,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def triplet_rows(
    triplets: list[models.Triplet],
    source: str,
) -> list[dict]:
    rows: list[dict] = []
    for triplet in triplets:
        for node_id in triplet.node_ids:
            rows.append(triplet_properties(triplet, source, node_id))
    return rows


def evidence_pairs(
    triplets: list[models.Triplet],
    source: str,
) -> list[dict]:
    pairs: list[dict] = []
    for triplet in triplets:
        for node_id in triplet.node_ids:
            pairs.append(
                {
                    'node': nodes.node_uuid(source, node_id),
                    'triplet': triplet_uuid(
                        source,
                        node_id,
                        triplet.subject,
                        triplet.predicate,
                        triplet.object,
                    ),
                }
            )
    return pairs
