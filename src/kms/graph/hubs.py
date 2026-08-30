"""Source-local and cross-source canonical hub labels."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

LOCAL_ENTITY_HUB_LABEL = 'LocalEntityHub'
LOCAL_EVENT_HUB_LABEL = 'LocalEventHub'
LOCAL_PREDICATE_HUB_LABEL = 'LocalPredicateHub'
GLOBAL_ENTITY_HUB_LABEL = 'GlobalEntityHub'
GLOBAL_EVENT_HUB_LABEL = 'GlobalEventHub'
GLOBAL_PREDICATE_HUB_LABEL = 'GlobalPredicateHub'
LOCAL_TRIPLET_HUB_LABEL = 'LocalTripletHub'
GLOBAL_TRIPLET_HUB_LABEL = 'GlobalTripletHub'
ENTITY_HUB_LABEL = LOCAL_ENTITY_HUB_LABEL
PREDICATE_HUB_LABEL = LOCAL_PREDICATE_HUB_LABEL
META_ENTITY_HUB_LABEL = GLOBAL_ENTITY_HUB_LABEL
META_PREDICATE_HUB_LABEL = GLOBAL_PREDICATE_HUB_LABEL
TRIPLET_HUB_LABEL = LOCAL_TRIPLET_HUB_LABEL
META_TRIPLET_HUB_LABEL = GLOBAL_TRIPLET_HUB_LABEL


_LOCAL_LABELS = {
    'entity': LOCAL_ENTITY_HUB_LABEL,
    'event': LOCAL_EVENT_HUB_LABEL,
    'predicate': LOCAL_PREDICATE_HUB_LABEL,
}
_GLOBAL_LABELS = {
    'entity': GLOBAL_ENTITY_HUB_LABEL,
    'event': GLOBAL_EVENT_HUB_LABEL,
    'predicate': GLOBAL_PREDICATE_HUB_LABEL,
}
_COMPONENT_LABELS = {
    'entity': 'Entity',
    'event': 'Event',
    'predicate': 'Predicate',
}


def component_label(kind: str) -> str:
    """Returns the durable component label for a hub-building kind."""
    try:
        return _COMPONENT_LABELS[kind]
    except KeyError as error:
        raise ValueError(f'unknown component kind: {kind}') from error


def hub_label(kind: str, tier: str) -> str:
    """Returns the Neo4j label for a hub kind and tier."""
    if tier in {'local', 'source'}:
        labels = _LOCAL_LABELS
    elif tier in {'global', 'meta'}:
        labels = _GLOBAL_LABELS
    else:
        raise ValueError(f'unknown hub tier: {tier}')
    try:
        return labels[kind]
    except KeyError as error:
        raise ValueError(f'unknown hub kind: {kind}') from error


def local_hub_label(kind: str) -> str:
    """Returns the local semantic hub label."""
    return hub_label(kind, tier='local')


def global_hub_label(kind: str) -> str:
    """Returns the global semantic hub label."""
    return hub_label(kind, tier='global')


def triplet_hub_label(tier: str) -> str:
    """Returns the local or global triplet hub label."""
    if tier in {'local', 'source'}:
        return LOCAL_TRIPLET_HUB_LABEL
    if tier in {'global', 'meta'}:
        return GLOBAL_TRIPLET_HUB_LABEL
    raise ValueError(f'unknown triplet hub tier: {tier}')


def local_triplet_hub_label() -> str:
    """Returns the local triplet hub label."""
    return LOCAL_TRIPLET_HUB_LABEL


def global_triplet_hub_label() -> str:
    """Returns the global triplet hub label."""
    return GLOBAL_TRIPLET_HUB_LABEL


def triplet_hub_uuid(
    tier: str,
    source: str | None,
    subject_hub: str,
    predicate_hub: str,
    object_hub: str,
) -> str:
    """Returns the stable id for an ordered canonical triplet tuple."""
    if tier not in {'local', 'source', 'global', 'meta'}:
        raise ValueError(f'unknown triplet hub tier: {tier}')
    is_local = tier in {'local', 'source'}
    scope = source if is_local else 'meta'
    return uuid5(
        NAMESPACE_URL,
        f'{scope}#triplet_hub#{subject_hub}#{predicate_hub}#{object_hub}',
    ).hex


def triplet_hub_properties(
    tier: str,
    source: str | None,
    canonical_name: str,
    description: str,
    embedding: list[float] | None,
    subject_hub: str,
    predicate_hub: str,
    object_hub: str,
    *,
    hub_id: str,
) -> dict:
    """Builds properties for a derived TripletHub node."""
    if tier in {'local', 'source'} and source is None:
        raise ValueError('local triplet hubs require a source')
    if tier in {'global', 'meta'} and source is not None:
        raise ValueError('global triplet hubs cannot have a source')
    properties = {
        'uuid': hub_id,
        'source': nodes.source_uuid(source) if source else None,
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
        'subject_hub': subject_hub,
        'predicate_hub': predicate_hub,
        'object_hub': object_hub,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def hub_uuid(kind: str, source: str, identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#{kind}_hub#{identity}').hex


def global_hub_uuid(kind: str, identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'meta#{kind}_hub#{identity}').hex


def hub_properties(
    kind: str,
    source: str | None,
    canonical_name: str,
    aliases: list[str],
    description: str,
    embedding: list[float] | None = None,
    *,
    tier: str,
    hub_id: str | None = None,
) -> dict:
    if tier == 'source':
        if source is None:
            raise ValueError('source hubs require a source')
        if hub_id is None:
            raise ValueError('source hubs require an explicit hub_id')
    elif tier == 'meta':
        if hub_id is None:
            raise ValueError('meta hubs require an explicit hub_id')
    else:
        raise ValueError(f'unknown hub tier: {tier}')
    properties = {
        'uuid': hub_id,
        'source': nodes.source_uuid(source) if source is not None else None,
        'canonical_name': canonical_name,
        'aliases': aliases,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
