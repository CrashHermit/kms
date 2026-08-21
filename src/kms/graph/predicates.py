"""Per-triplet predicate vertices carrying the relation text and description."""

from kms.core import identity
from kms.graph import nodes

PREDICATE_LABEL = 'Predicate'


def predicate_uuid(triplet_uuid: str) -> str:
    """Returns the deterministic uuid for a per-triplet predicate."""
    return identity.predicate_uuid(triplet_uuid)


def predicate_properties(
    source: str,
    node_position: int,
    triplet_uuid: str,
    predicate: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """Builds the property dict used to persist a per-triplet predicate."""
    properties = {
        'uuid': predicate_uuid(triplet_uuid),
        'source': nodes.source_uuid(source),
        'node_position': node_position,
        'predicate': predicate,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
