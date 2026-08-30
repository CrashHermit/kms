"""EntityHub graph labels, ids, and property builders."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

COMPONENT_LABEL = 'Entity'
LOCAL_HUB_LABEL = 'LocalEntityHub'
GLOBAL_HUB_LABEL = 'GlobalEntityHub'
LOCAL_HUB_LABEL = LOCAL_HUB_LABEL


def component_label() -> str:
    return COMPONENT_LABEL


def hub_label(tier: str = 'local') -> str:
    if tier in {'local', 'source'}:
        return LOCAL_HUB_LABEL
    if tier in {'global', 'meta'}:
        return GLOBAL_HUB_LABEL
    raise ValueError(f'unknown hub tier: {tier}')


def hub_uuid(source: str, identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#entity_hub#{identity}').hex


def local_hub_uuid(source: str, identity: str) -> str:
    """Returns the deterministic local entity-hub UUID."""
    return hub_uuid(source, identity)


def global_hub_uuid(identity: str) -> str:
    """Returns the deterministic global entity-hub UUID."""
    return uuid5(NAMESPACE_URL, f'meta#entity_hub#{identity}').hex




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
    if tier in {'local', 'source'}:
        if source is None:
            raise ValueError('local entity hubs require a source')
        if hub_id is None:
            raise ValueError('local entity hubs require an explicit hub_id')
    elif tier in {'global', 'meta'}:
        if hub_id is None:
            raise ValueError('global entity hubs require an explicit hub_id')
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
