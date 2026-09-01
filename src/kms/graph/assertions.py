"""Row and edge builders for the assertion layer.

The assertion layer is the durable tier between :Triplet and the hubs:
each triplet occurrence yields node-local :Entity or :Event vertices for
its subject and object plus a per-triplet :Predicate vertex.
"""

from kms.core import identity, models
from kms.graph import entities, events, names, predicates


def _endpoint(
    kind: models.NodeKind,
    source: str,
    node_position: int,
    name: str,
    description: str | None,
    embedding: list[float] | None,
) -> tuple[str, dict]:
    """Build one typed assertion endpoint row."""
    if kind is models.NodeKind.ENTITY:
        return entities.entity_uuid(
            source, node_position, name
        ), entities.entity_properties(
            source, node_position, name, description, embedding
        )
    if kind is models.NodeKind.EVENT:
        return events.event_uuid(
            source, node_position, name
        ), events.event_properties(
            source, node_position, name, description, embedding
        )
    raise ValueError(f'unknown triplet endpoint kind: {kind}')


def assertion_rows(
    triplets: list[models.Triplet],
    source: str,
    entity_descriptions: dict[int, dict[str, str | None]],
    predicate_descriptions: dict[int, dict[str, str | None]],
    entity_embeddings: dict[int, dict[str, list[float]]] | None = None,
    predicate_embeddings: dict[int, dict[str, list[float]]] | None = None,
    event_descriptions: dict[int, dict[str, str | None]] | None = None,
    event_embeddings: dict[int, dict[str, list[float]]] | None = None,
) -> dict:
    """Builds entity/event/predicate rows and edge pairs for assertions.

    Args:
        triplets: The extracted triplets.
        source: The raw source key.
        entity_descriptions: Per-node entity descriptions.
        predicate_descriptions: Per-node predicate descriptions.
        entity_embeddings: Per-node entity embeddings.
        predicate_embeddings: Per-node predicate embeddings.
        event_descriptions: Per-node event descriptions.
        event_embeddings: Per-node event embeddings.

    Returns:
        A dict with typed component rows, names, and assertion edge lists.
    """
    entity_rows: dict[str, dict] = {}
    event_rows: dict[str, dict] = {}
    predicate_rows: dict[str, dict] = {}
    entity_name_rows: dict[str, dict] = {}
    event_name_rows: dict[str, dict] = {}
    predicate_name_rows: dict[str, dict] = {}
    entity_name_edges: list[dict] = []
    event_name_edges: list[dict] = []
    predicate_name_edges: list[dict] = []
    subject_edges: list[dict] = []
    object_edges: list[dict] = []
    predicate_edges: list[dict] = []

    for triplet in triplets:
        for node_position in triplet.evidence_positions:
            triplet_id = triplet.occurrence_uuids.get(node_position)
            expected_triplet_id = identity.triplet_uuid(
                source,
                node_position,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            if triplet_id is None:
                raise ValueError(
                    f'triplet occurrence for position {node_position} is missing its uuid'
                )
            if triplet_id != expected_triplet_id:
                raise ValueError(
                    f'triplet occurrence uuid {triplet_id!r} does not match '
                    f'expected {expected_triplet_id!r}'
                )
            subject_descriptions = (
                event_descriptions
                if triplet.subject_kind is models.NodeKind.EVENT
                else entity_descriptions
            )
            subject_embeddings = (
                event_embeddings
                if triplet.subject_kind is models.NodeKind.EVENT
                else entity_embeddings
            )
            subject_id, subject_row = _endpoint(
                triplet.subject_kind,
                source,
                node_position,
                triplet.subject,
                (subject_descriptions or {})
                .get(node_position, {})
                .get(triplet.subject),
                (subject_embeddings or {})
                .get(node_position, {})
                .get(triplet.subject),
            )
            object_descriptions = (
                event_descriptions
                if triplet.object_kind is models.NodeKind.EVENT
                else entity_descriptions
            )
            object_embeddings = (
                event_embeddings
                if triplet.object_kind is models.NodeKind.EVENT
                else entity_embeddings
            )
            object_id, object_row = _endpoint(
                triplet.object_kind,
                source,
                node_position,
                triplet.object,
                (object_descriptions or {})
                .get(node_position, {})
                .get(triplet.object),
                (object_embeddings or {})
                .get(node_position, {})
                .get(triplet.object),
            )
            expected_subject_id = subject_id
            expected_object_id = object_id
            expected_predicate_id = predicates.predicate_uuid(triplet_id)
            uuid_map = (
                triplet.entity_uuids
                if triplet.subject_kind is models.NodeKind.ENTITY
                else triplet.event_uuids
            )
            subject_id = uuid_map.get(
                (node_position, triplet.subject), subject_id
            )
            uuid_map = (
                triplet.entity_uuids
                if triplet.object_kind is models.NodeKind.ENTITY
                else triplet.event_uuids
            )
            object_id = uuid_map.get((node_position, triplet.object), object_id)
            predicate_id = triplet.predicate_uuids.get(
                node_position, expected_predicate_id
            )
            if subject_id != expected_subject_id:
                raise ValueError('triplet subject endpoint uuid does not match')
            if object_id != expected_object_id:
                raise ValueError('triplet object endpoint uuid does not match')
            if predicate_id != expected_predicate_id:
                raise ValueError('triplet predicate uuid does not match')
            endpoint_rows = (
                entity_rows
                if triplet.subject_kind is models.NodeKind.ENTITY
                else event_rows
            )
            endpoint_rows[subject_id] = subject_row
            endpoint_rows = (
                entity_rows
                if triplet.object_kind is models.NodeKind.ENTITY
                else event_rows
            )
            endpoint_rows[object_id] = object_row
            subject_name_kind = triplet.subject_kind.value
            object_name_kind = triplet.object_kind.value
            subject_name_id = names.name_uuid(subject_name_kind, subject_id)
            object_name_id = names.name_uuid(object_name_kind, object_id)
            subject_name_rows = (
                entity_name_rows
                if triplet.subject_kind is models.NodeKind.ENTITY
                else event_name_rows
            )
            object_name_rows = (
                entity_name_rows
                if triplet.object_kind is models.NodeKind.ENTITY
                else event_name_rows
            )
            subject_name_rows[subject_name_id] = names.name_properties(
                subject_name_kind,
                source,
                subject_id,
                triplet.subject,
                node_position,
            )
            object_name_rows[object_name_id] = names.name_properties(
                object_name_kind,
                source,
                object_id,
                triplet.object,
                node_position,
            )
            subject_name_edges = (
                entity_name_edges
                if triplet.subject_kind is models.NodeKind.ENTITY
                else event_name_edges
            )
            object_name_edges = (
                entity_name_edges
                if triplet.object_kind is models.NodeKind.ENTITY
                else event_name_edges
            )
            subject_name_edges.append(
                {'component': subject_id, 'name': subject_name_id}
            )
            object_name_edges.append(
                {'component': object_id, 'name': object_name_id}
            )
            node_predicate_embeddings = (predicate_embeddings or {}).get(
                node_position, {}
            )
            predicate_name_id = names.name_uuid('predicate', predicate_id)
            predicate_name_rows[predicate_name_id] = names.name_properties(
                'predicate',
                source,
                predicate_id,
                triplet.predicate,
                node_position,
            )
            predicate_name_edges.append(
                {'component': predicate_id, 'name': predicate_name_id}
            )
            predicate_rows[predicate_id] = predicates.predicate_properties(
                source,
                node_position,
                triplet_id,
                triplet.predicate,
                predicate_descriptions.get(node_position, {}).get(
                    triplet.predicate
                ),
                node_predicate_embeddings.get(triplet.predicate),
            )

            subject_edges.append({'triplet': triplet_id, 'entity': subject_id})
            object_edges.append({'triplet': triplet_id, 'entity': object_id})
            predicate_edges.append(
                {'triplet': triplet_id, 'predicate': predicate_id}
            )

    return {
        'events': list(event_rows.values()),
        'entities': list(entity_rows.values()),
        'predicates': list(predicate_rows.values()),
        'entity_names': list(entity_name_rows.values()),
        'event_names': list(event_name_rows.values()),
        'predicate_names': list(predicate_name_rows.values()),
        'entity_name_edges': entity_name_edges,
        'event_name_edges': event_name_edges,
        'predicate_name_edges': predicate_name_edges,
        'subject_edges': subject_edges,
        'object_edges': object_edges,
        'predicate_edges': predicate_edges,
    }
