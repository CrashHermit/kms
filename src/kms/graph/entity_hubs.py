"""
Graph representation of the entity canonicalization layer —
``:EntityHub`` vertices acting as cluster anchors.

The canonicalization pass clusters entity spokes by embedding similarity
and writes one ``:EntityHub`` per cluster. The hub is empty — it carries
only its uuid and source. The content lives on the ``:Definition`` vertex
reached via ``(:EntityHub)-[:HAS_DEFINITION]->(:Definition)``.

Identity: deterministic uuid5 over ``(source, sorted(spoke_uuids))`` —
the same set of spokes always produces the same hub uuid, so
re-clustering is idempotent.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_HUB_LABEL = 'EntityHub'


def entity_hub_uuid(source: str, spoke_uuids: list[str]) -> str:
    """Stable, deterministic vertex key for one entity cluster.

    Args:
        source: The stable book identity.
        spoke_uuids: The uuids of every entity in the cluster (order
            does not matter — they are sorted internally).

    Returns:
        The hub's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#entity_hub#{nodes.block_key(sorted(spoke_uuids))}',
    ).hex


def entity_hub_properties(source: str, spoke_uuids: list[str]) -> dict:
    """The Neo4j property map for one entity hub.

    The hub is empty — content lives on ``:Definition``.

    Args:
        source: The stable book identity.
        spoke_uuids: The cluster's spoke uuids, used (sorted) for
            identity.

    Returns:
        The property map.
    """
    return {
        'uuid': entity_hub_uuid(source, spoke_uuids),
        'source': nodes.source_uuid(source),
    }


def entity_hub_rows(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    """Every entity hub's property map, one flat list.

    Args:
        clusters: One list of spoke dicts per cluster. Each spoke dict
            carries at least ``uuid``.
        source: The stable book identity.

    Returns:
        One property map per cluster.
    """
    return [
        entity_hub_properties(
            source, [spoke['uuid'] for spoke in cluster]
        )
        for cluster in clusters
    ]


def canonical_entity_pairs(
    clusters: list[list[dict]], source: str
) -> list[dict]:
    """The ``{entity, hub}`` uuid pairs for ``:CANONICAL`` edges.

    Every spoke points at its cluster's hub via ``:CANONICAL``.

    Args:
        clusters: One list of spoke dicts per cluster.
        source: The stable book identity.

    Returns:
        One ``{entity, hub}`` per spoke.
    """
    pairs: list[dict] = []
    for cluster in clusters:
        spoke_uuids = [spoke['uuid'] for spoke in cluster]
        hub_uuid = entity_hub_uuid(source, spoke_uuids)
        for spoke in cluster:
            pairs.append(
                {'entity': spoke['uuid'], 'hub': hub_uuid}
            )
    return pairs
