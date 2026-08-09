"""
Unified hub mapping for entity and predicate canonicalization.

:EntityHub and :PredicateHub are structurally identical empty connectors.
The only difference is the Neo4j label.  This module handles both through
a ``kind`` parameter.

Identity: deterministic uuid5 over ``(source, sorted(spoke_uuids))`` —
idempotent for a given spoke set.  Stability across rebuilds that add new
sources is achieved by old-to-new hub matching via spoke overlap before
the canonical layer is replaced (see ``canonicalizer._match_old_hubs``).

Replaces ``entity_hubs`` and ``predicate_hubs``, which were structurally
identical copies of each other.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_HUB_LABEL = 'EntityHub'
PREDICATE_HUB_LABEL = 'PredicateHub'

HUB_LABELS = {'entity': ENTITY_HUB_LABEL, 'predicate': PREDICATE_HUB_LABEL}


def hub_uuid(source: str, spoke_uuids: list[str], kind: str) -> str:
    """Deterministic vertex key for one cluster.

    Args:
        source: The stable book identity (majority source for the cluster).
        spoke_uuids: Every spoke uuid in the cluster, sorted internally.
        kind: ``'entity'`` or ``'predicate'``.

    Returns:
        The hub's hex uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#{kind}_hub#{nodes.block_key(sorted(spoke_uuids))}',
    ).hex


def hub_properties(
    source: str,
    spoke_uuids: list[str],
    kind: str,
    *,
    display_name: str | None = None,
    aliases: list[str] | None = None,
) -> dict:
    """The Neo4j property map for one hub.

    Args:
        source: The stable book identity.
        spoke_uuids: The cluster's spoke uuids, used (sorted) for identity.
        kind: ``'entity'`` or ``'predicate'``.
        display_name: The most frequent surface form in the cluster.
        aliases: Alternative surface forms merged into this hub.

    Returns:
        The property map, with None/empty values omitted.
    """
    properties = {
        'uuid': hub_uuid(source, spoke_uuids, kind),
        'source': nodes.source_uuid(source),
        'display_name': display_name,
        'aliases': aliases or [],
    }
    return {
        key: value
        for key, value in properties.items()
        if value is not None and value != []
    }


def hub_rows(
    clusters: list[list[dict]],
    source: str,
    kind: str,
    definitions: list[dict] | None = None,
) -> list[dict]:
    """Every hub's property map, one flat list.

    Args:
        clusters: One list of spoke dicts per cluster.
        source: The stable book identity.
        kind: ``'entity'`` or ``'predicate'``.
        definitions: Optional definition dicts (aligned with
            ``clusters``) carrying ``display_name``.

    Returns:
        One property map per cluster.
    """
    rows: list[dict] = []
    for i, cluster in enumerate(clusters):
        spoke_uuids = [s['uuid'] for s in cluster]
        display_name = (
            definitions[i].get('display_name')
            if definitions and i < len(definitions)
            else None
        )
        rows.append(
            hub_properties(
                source, spoke_uuids, kind, display_name=display_name
            )
        )
    return rows


def canonical_pairs(
    clusters: list[list[dict]],
    kind: str,
) -> list[dict]:
    """The ``{spoke, hub}`` uuid pairs for ``:CANONICAL`` edges.

    Source is computed per cluster from the majority spoke source.

    Args:
        clusters: One list of spoke dicts per cluster.
        kind: ``'entity'`` → key ``'entity'``, ``'predicate'`` → key
            ``'predicate'``.

    Returns:
        One ``{entity_or_predicate, hub}`` per spoke.
    """
    from collections import Counter

    key = 'entity' if kind == 'entity' else 'predicate'
    pairs: list[dict] = []
    for cluster in clusters:
        spoke_uuids = [s['uuid'] for s in cluster]
        sources = [s.get('source', 'unknown') for s in cluster]
        source = Counter(sources).most_common(1)[0][0]
        hub = hub_uuid(source, spoke_uuids, kind)
        for spoke in cluster:
            pairs.append({key: spoke['uuid'], 'hub': hub})
    return pairs
