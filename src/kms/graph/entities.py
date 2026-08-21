"""Node-local entity vertices that hang off each triplet occurrence."""

from kms.core import identity
from kms.graph import nodes

ENTITY_LABEL = 'Entity'


def entity_uuid(source: str, node_position: int, name: str) -> str:
    """Returns the deterministic uuid for a node-local entity.

    Identity is local per node: the same surface form at a different
    node is a separate vertex, so re-derivation is idempotent.
    """
    return identity.entity_uuid(source, node_position, name)


def entity_properties(
    source: str,
    node_position: int,
    name: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """Builds the property dict used to persist a node-local entity."""
    properties = {
        'uuid': entity_uuid(source, node_position, name),
        'source': nodes.source_uuid(source),
        'node_position': node_position,
        'name': name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
