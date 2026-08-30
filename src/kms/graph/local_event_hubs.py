"""EventHub graph labels, IDs, and property builders."""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

COMPONENT_LABEL = 'Event'
LOCAL_HUB_LABEL = 'LocalEventHub'
GLOBAL_HUB_LABEL = 'GlobalEventHub'


def component_label() -> str:
    return COMPONENT_LABEL


def hub_label(tier: str = 'local') -> str:
    if tier in {'local', 'source'}:
        return LOCAL_HUB_LABEL
    if tier in {'global', 'meta'}:
        return GLOBAL_HUB_LABEL
    raise ValueError(f'unknown hub tier: {tier}')


def hub_uuid(source: str, identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#event_hub#{identity}').hex


def local_hub_uuid(source: str, identity: str) -> str:
    return hub_uuid(source, identity)


def global_hub_uuid(identity: str) -> str:
    return uuid5(NAMESPACE_URL, f'meta#event_hub#{identity}').hex


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
    if hub_id is None:
        raise ValueError('event hubs require an explicit hub_id')
    if tier in {'local', 'source'} and source is None:
        raise ValueError('local event hubs require a source')
    if tier not in {'local', 'source', 'global', 'meta'}:
        raise ValueError(f'unknown event hub tier: {tier}')
    properties = {
        'uuid': hub_id,
        'source': nodes.source_uuid(source) if source else None,
        'canonical_name': canonical_name,
        'aliases': aliases,
        'description': description,
        'embedding': embedding,
    }
    return {key: value for key, value in properties.items() if value is not None}
