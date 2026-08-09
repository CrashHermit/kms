from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.entities import entity_uuid
from kms.graph.facts import fact_uuid

TRIPLET_LABEL = 'Triplet'


def triplet_uuid(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#triplet#{node_id}#{fact_uuid_val}'
        f'#{subject}#{predicate}#{object}',
    ).hex


def triplet_properties(
    triplet: models.Triplet,
    source: str,
    node_id: int,
    fact_uuid_val: str,
) -> dict:
    properties = {
        'uuid': triplet_uuid(
            source,
            node_id,
            fact_uuid_val,
            triplet.subject,
            triplet.predicate,
            triplet.object,
        ),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'fact_index': triplet.fact_index,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _triplet_nodes(
    triplet: models.Triplet, facts: list[models.AtomicFact]
) -> list[int]:
    fact = facts[triplet.fact_index]
    return list(fact.node_ids)


def triplet_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    rows: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            rows.append(
                triplet_properties(triplet, source, node_id, fact_uuid_val)
            )
    return rows


def yields_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            pairs.append(
                {
                    'fact': fact_uuid_val,
                    'triplet': triplet_uuid(
                        source,
                        node_id,
                        fact_uuid_val,
                        triplet.subject,
                        triplet.predicate,
                        triplet.object,
                    ),
                }
            )
    return pairs


def _role_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    *,
    role: str,
) -> list[dict]:
    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            triplet_uuid_val = triplet_uuid(
                source,
                node_id,
                fact_uuid_val,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            pairs.append(
                {
                    'triplet': triplet_uuid_val,
                    'entity': entity_uuid(
                        source, node_id, triplet_uuid_val, role
                    ),
                }
            )
    return pairs


def has_subject_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    return _role_pairs(
        triplets,
        facts,
        source,
        role='subject',
    )


def has_object_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    return _role_pairs(
        triplets,
        facts,
        source,
        role='object',
    )

