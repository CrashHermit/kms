"""Source-local and cross-source canonical hub labels."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_HUB_LABEL = 'EntityHub'
PREDICATE_HUB_LABEL = 'PredicateHub'
META_ENTITY_HUB_LABEL = 'MetaEntityHub'
META_PREDICATE_HUB_LABEL = 'MetaPredicateHub'
TRIPLET_HUB_LABEL = 'TripletHub'
META_TRIPLET_HUB_LABEL = 'MetaTripletHub'

_SOURCE_LABELS = {
    'entity': ENTITY_HUB_LABEL,
    'predicate': PREDICATE_HUB_LABEL,
}
_META_LABELS = {
    'entity': META_ENTITY_HUB_LABEL,
    'predicate': META_PREDICATE_HUB_LABEL,
}
_COMPONENT_LABELS = {
    'entity': 'Entity',
    'predicate': 'Predicate',
}


def component_label(kind: str) -> str:
    """Returns the durable component label for a canonicalization kind."""
    try:
        return _COMPONENT_LABELS[kind]
    except KeyError as error:
        raise ValueError(f'unknown component kind: {kind}') from error


def hub_label(kind: str, tier: str) -> str:
    """Returns the Neo4j label for a hub kind and tier.

    The ``source`` tier contains source-local hubs. The ``meta`` tier is the
    disposable cross-source hub-over-hubs layer.
    """
    if tier == 'source':
        labels = _SOURCE_LABELS
    elif tier == 'meta':
        labels = _META_LABELS
    else:
        raise ValueError(f'unknown hub tier: {tier}')
    return labels[kind]


def meta_hub_label(kind: str) -> str:
    """Returns the Neo4j label for a cross-source meta hub."""
    return hub_label(kind, tier='meta')


def triplet_hub_label(tier: str) -> str:
    """Returns the label for a source-local or cross-source triplet hub."""
    if tier == 'source':
        return TRIPLET_HUB_LABEL
    if tier == 'meta':
        return META_TRIPLET_HUB_LABEL
    raise ValueError(f'unknown triplet hub tier: {tier}')


def triplet_hub_uuid(
    tier: str,
    source: str | None,
    subject_hub: str,
    predicate_hub: str,
    object_hub: str,
) -> str:
    """Returns the stable id for an ordered canonical triplet tuple."""
    if tier not in {'source', 'meta'}:
        raise ValueError(f'unknown triplet hub tier: {tier}')
    scope = source if tier == 'source' else 'meta'
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
    if tier == 'source' and source is None:
        raise ValueError('source triplet hubs require a source')
    if tier == 'meta' and source is not None:
        raise ValueError('meta triplet hubs cannot have a source')
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
    """Returns the deterministic uuid for a source-local hub identity."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#{kind}_hub#{identity}',
    ).hex


def meta_hub_uuid(kind: str, identity: str) -> str:
    """Returns a stable uuid for a meta hub identity.

    ``identity`` is supplied by the meta canonicalizer and is based on
    source-hub membership rather than a mutable synthesized canonical name.
    """
    return uuid5(
        NAMESPACE_URL,
        f'meta#{kind}_hub#{identity}',
    ).hex


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
    """Builds a source or meta hub property dict.

    Source hubs derive an id when ``hub_id`` is not supplied. Meta hubs
    require an explicit stable id because their source membership and
    synthesized name can change independently.
    """
    if tier == 'source':
        if source is None:
            raise ValueError('source hubs require a source')
        if hub_id is None:
            raise ValueError('source hubs require an explicit hub_id')
        uuid = hub_id
    elif tier == 'meta':
        if hub_id is None:
            raise ValueError('meta hubs require an explicit hub_id')
        uuid = hub_id
    else:
        raise ValueError(f'unknown hub tier: {tier}')

    properties = {
        'uuid': uuid,
        'source': nodes.source_uuid(source) if source is not None else None,
        'canonical_name': canonical_name,
        'aliases': aliases,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
