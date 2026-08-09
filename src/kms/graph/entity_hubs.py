from kms.graph.hubs import (
    ENTITY_HUB_LABEL,
)
from kms.graph.hubs import canonical_pairs as _cp
from kms.graph.hubs import (
    hub_properties as entity_hub_properties,
)
from kms.graph.hubs import (
    hub_rows as entity_hub_rows,
)
from kms.graph.hubs import (
    hub_uuid as entity_hub_uuid,
)


def canonical_entity_pairs(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    return _cp(clusters, 'entity')


__all__ = [
    'ENTITY_HUB_LABEL',
    'canonical_entity_pairs',
    'entity_hub_properties',
    'entity_hub_rows',
    'entity_hub_uuid',
]

