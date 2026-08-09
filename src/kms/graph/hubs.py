from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_HUB_LABEL = 'EntityHub'
PREDICATE_HUB_LABEL = 'PredicateHub'

HUB_LABELS = {'entity': ENTITY_HUB_LABEL, 'predicate': PREDICATE_HUB_LABEL}


def hub_uuid(source: str, spoke_uuids: list[str], kind: str) -> str:
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

