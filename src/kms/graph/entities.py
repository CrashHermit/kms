from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.facts import fact_uuid

ENTITY_LABEL = 'Entity'


def entity_uuid(
    source: str, node_id: int, triplet_uuid_val: str, role: str
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#entity#{node_id}#{triplet_uuid_val}#{role}',
    ).hex


def entity_properties(
    source: str,
    node_id: int,
    triplet_uuid_val: str,
    role: str,
    name: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': entity_uuid(source, node_id, triplet_uuid_val, role),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'role': role,
        'name': name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _description_at(
    node_id: int,
    name: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> str | None:
    for entry in node_entity_descriptions.get(node_id, []):
        if entry['name'] == name:
            return entry.get('description')
    return None


def _embedding_at(
    node_id: int,
    name: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> list[float] | None:
    for entry in node_entity_descriptions.get(node_id, []):
        if entry['name'] == name:
            return entry.get('embedding')
    return None


def _triplet_nodes(
    triplet: models.Triplet, facts: list[models.AtomicFact]
) -> list[int]:
    fact = facts[triplet.fact_index]
    return list(fact.node_ids)


def entity_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> list[dict]:
    from kms.graph.triplets import triplet_uuid

    rows: list[dict] = []
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
            for role, name in (
                ('subject', triplet.subject),
                ('object', triplet.object),
            ):
                description = _description_at(
                    node_id, name, node_entity_descriptions
                )
                embedding = _embedding_at(
                    node_id, name, node_entity_descriptions
                )
                rows.append(
                    entity_properties(
                        source,
                        node_id,
                        triplet_uuid_val,
                        role,
                        name,
                        description,
                        embedding,
                    )
                )
    return rows

