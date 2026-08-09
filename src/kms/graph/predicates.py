from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.facts import fact_uuid

PREDICATE_LABEL = 'Predicate'


def predicate_uuid(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#predicate#{node_id}#{fact_uuid_val}'
        f'#{subject}#{predicate}#{object}',
    ).hex


def predicate_properties(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': predicate_uuid(
            source, node_id, fact_uuid_val, subject, predicate, object
        ),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'predicate': predicate,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _description_for(
    node_id: int,
    predicate: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> str | None:
    for entry in node_predicate_descriptions.get(node_id, []):
        if entry['predicate'] == predicate:
            return entry.get('description')
    return None


def _embedding_for(
    node_id: int,
    predicate: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> list[float] | None:
    for entry in node_predicate_descriptions.get(node_id, []):
        if entry['predicate'] == predicate:
            return entry.get('embedding')
    return None


def _triplet_nodes(
    triplet: models.Triplet, facts: list[models.AtomicFact]
) -> list[int]:
    fact = facts[triplet.fact_index]
    return list(fact.node_ids)


def predicate_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> list[dict]:
    rows: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            description = _description_for(
                node_id, triplet.predicate, node_predicate_descriptions
            )
            embedding = _embedding_for(
                node_id, triplet.predicate, node_predicate_descriptions
            )
            rows.append(
                predicate_properties(
                    source,
                    node_id,
                    fact_uuid_val,
                    triplet.subject,
                    triplet.predicate,
                    triplet.object,
                    description,
                    embedding,
                )
            )
    return rows


def has_predicate_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    from kms.graph.triplets import triplet_uuid

    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            pairs.append(
                {
                    'triplet': triplet_uuid(
                        source,
                        node_id,
                        fact_uuid_val,
                        triplet.subject,
                        triplet.predicate,
                        triplet.object,
                    ),
                    'predicate': predicate_uuid(
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

