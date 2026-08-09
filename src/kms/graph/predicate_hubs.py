from kms.graph.hubs import (
    PREDICATE_HUB_LABEL,
)
from kms.graph.hubs import canonical_pairs as _cp
from kms.graph.hubs import (
    hub_properties as predicate_hub_properties,
)
from kms.graph.hubs import (
    hub_rows as predicate_hub_rows,
)
from kms.graph.hubs import (
    hub_uuid as predicate_hub_uuid,
)


def canonical_predicate_pairs(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    return _cp(clusters, 'predicate')


__all__ = [
    'PREDICATE_HUB_LABEL',
    'canonical_predicate_pairs',
    'predicate_hub_properties',
    'predicate_hub_rows',
    'predicate_hub_uuid',
]

