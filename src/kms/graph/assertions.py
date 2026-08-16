"""Row and edge builders for the assertion layer.

The assertion layer is the durable tier between :Triplet and the hubs:
each triplet occurrence yields node-local :Entity vertices for its
subject and object plus a per-triplet :Predicate vertex, each described
from the window around its source node.
"""

from kms.core import models
from kms.graph import entities, names, predicates
from kms.graph.triplets import triplet_uuid


def assertion_rows(
    triplets: list[models.Triplet],
    source: str,
    entity_descriptions: dict[int, dict[str, str | None]],
    predicate_descriptions: dict[int, dict[str, str | None]],
    entity_embeddings: dict[int, dict[str, list[float]]] | None = None,
    predicate_embeddings: dict[int, dict[str, list[float]]] | None = None,
) -> dict:
    """Builds entity/predicate rows and edge pairs for the assertion layer.

    Args:
        triplets: The extracted triplets.
        source: The raw source key.
        entity_descriptions: Per-node map of entity name to description.
        predicate_descriptions: Per-node map of predicate text to
            description.
        entity_embeddings: Per-node map of entity name to embedding.
        predicate_embeddings: Per-node map of predicate text to embedding.

    Returns:
        A dict with 'entities', 'predicates', 'subject_edges',
        'object_edges', and 'predicate_edges' lists.
    """
    entity_rows: dict[str, dict] = {}
    predicate_rows: dict[str, dict] = {}
    entity_name_rows: dict[str, dict] = {}
    predicate_name_rows: dict[str, dict] = {}
    entity_name_edges: list[dict] = []
    predicate_name_edges: list[dict] = []
    subject_edges: list[dict] = []
    object_edges: list[dict] = []
    predicate_edges: list[dict] = []

    for triplet in triplets:
        for node_id in triplet.node_ids:
            triplet_id = triplet_uuid(
                source,
                node_id,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            subject_id = entities.entity_uuid(source, node_id, triplet.subject)
            object_id = entities.entity_uuid(source, node_id, triplet.object)
            predicate_id = predicates.predicate_uuid(triplet_id)

            node_descriptions = entity_descriptions.get(node_id, {})
            node_entity_embeddings = (
                entity_embeddings.get(node_id, {}) if entity_embeddings else {}
            )
            node_predicate_embeddings = (
                predicate_embeddings.get(node_id, {})
                if predicate_embeddings
                else {}
            )
            entity_rows[subject_id] = entities.entity_properties(
                source,
                node_id,
                triplet.subject,
                node_descriptions.get(triplet.subject),
                node_entity_embeddings.get(triplet.subject),
            )
            entity_rows[object_id] = entities.entity_properties(
                source,
                node_id,
                triplet.object,
                node_descriptions.get(triplet.object),
                node_entity_embeddings.get(triplet.object),
            )
            subject_name_id = names.name_uuid('entity', subject_id)
            object_name_id = names.name_uuid('entity', object_id)
            entity_name_rows[subject_name_id] = names.name_properties(
                'entity', source, subject_id, triplet.subject, node_id
            )
            entity_name_rows[object_name_id] = names.name_properties(
                'entity', source, object_id, triplet.object, node_id
            )
            entity_name_edges.extend(
                [
                    {'component': subject_id, 'name': subject_name_id},
                    {'component': object_id, 'name': object_name_id},
                ]
            )
            predicate_name_id = names.name_uuid('predicate', predicate_id)
            predicate_name_rows[predicate_name_id] = names.name_properties(
                'predicate', source, predicate_id, triplet.predicate, node_id
            )
            predicate_name_edges.append(
                {'component': predicate_id, 'name': predicate_name_id}
            )
            predicate_rows[predicate_id] = predicates.predicate_properties(
                source,
                node_id,
                triplet_id,
                triplet.predicate,
                predicate_descriptions.get(node_id, {}).get(triplet.predicate),
                node_predicate_embeddings.get(triplet.predicate),
            )

            subject_edges.append({'triplet': triplet_id, 'entity': subject_id})
            object_edges.append({'triplet': triplet_id, 'entity': object_id})
            predicate_edges.append(
                {'triplet': triplet_id, 'predicate': predicate_id}
            )

    return {
        'entities': list(entity_rows.values()),
        'predicates': list(predicate_rows.values()),
        'entity_names': list(entity_name_rows.values()),
        'predicate_names': list(predicate_name_rows.values()),
        'entity_name_edges': entity_name_edges,
        'predicate_name_edges': predicate_name_edges,
        'subject_edges': subject_edges,
        'object_edges': object_edges,
        'predicate_edges': predicate_edges,
    }
