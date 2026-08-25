"""PredicateHub graph labels, ids, and property builders."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

COMPONENT_LABEL = 'Predicate'
LOCAL_HUB_LABEL = 'LocalPredicateHub'
GLOBAL_HUB_LABEL = 'GlobalPredicateHub'
HUB_LABEL = LOCAL_HUB_LABEL
META_HUB_LABEL = GLOBAL_HUB_LABEL


def component_label() -> str:
    return COMPONENT_LABEL


def hub_label(tier: str = 'source') -> str:
    if tier == 'source':
        return HUB_LABEL
    if tier == 'meta':
        return META_HUB_LABEL
    raise ValueError(f'unknown hub tier: {tier}')


def hub_uuid(source: str, identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#predicate_hub#{identity}').hex


def meta_hub_uuid(identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'meta#predicate_hub#{identity}').hex


def hub_properties(
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
            raise ValueError('source predicate hubs require a source')
        if hub_id is None:
            raise ValueError('source predicate hubs require an explicit hub_id')
    elif tier == 'meta':
        if hub_id is None:
            raise ValueError('meta predicate hubs require an explicit hub_id')
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
