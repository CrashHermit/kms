"""Re-exported from ``kms.graph.hubs`` — use that module directly."""

from kms.graph.hubs import (  # noqa: F401
    ENTITY_HUB_LABEL,
    canonical_pairs as _cp,
    hub_properties as entity_hub_properties,
    hub_rows as entity_hub_rows,
    hub_uuid as entity_hub_uuid,
)


def canonical_entity_pairs(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    """``{entity, hub}`` pairs via ``hubs.canonical_pairs``."""
    return _cp(clusters, 'entity')
