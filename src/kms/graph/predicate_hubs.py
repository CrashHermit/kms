"""Re-exported from ``kms.graph.hubs`` — use that module directly."""

from kms.graph.hubs import (  # noqa: F401
    PREDICATE_HUB_LABEL,
    canonical_pairs as _cp,
    hub_properties as predicate_hub_properties,
    hub_rows as predicate_hub_rows,
    hub_uuid as predicate_hub_uuid,
)


def canonical_predicate_pairs(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    """``{predicate, hub}`` pairs via ``hubs.canonical_pairs``."""
    return _cp(clusters, 'predicate')
