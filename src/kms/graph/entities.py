"""Node-local entity vertices that hang off each triplet occurrence."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_LABEL = 'Entity'


def entity_uuid(source: str, node_id: int, name: str) -> str:
    """Returns the deterministic uuid for a node-local entity.

    Identity is local per node: the same surface form at a different
    node is a separate vertex, so re-derivation is idempotent.
    """
    return uuid5(NAMESPACE_URL, f'{source}#entity#{node_id}#{name}').hex


def entity_properties(
    source: str,
    node_id: int,
    name: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """Builds the property dict used to persist a node-local entity."""
    properties = {
        'uuid': entity_uuid(source, node_id, name),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'name': name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
