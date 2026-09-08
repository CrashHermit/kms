"""Node-local event vertices that hang off triplet occurrences."""

from kms.core import identity
from kms.graph import nodes

EVENT_LABEL = 'Event'


def event_uuid(source: str, node_position: int, name: str) -> str:
    """Return the deterministic UUID for one node-local event."""
    return identity.event_uuid(source, node_position, name)


def event_properties(
    source: str,
    node_position: int,
    name: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """Build the property dict for one node-local event."""
    properties = {
        'uuid': event_uuid(source, node_position, name),
        'source': nodes.source_uuid(source),
        'node_position': node_position,
        'name': name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
