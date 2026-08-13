from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_HUB_LABEL = 'EntityHub'


def entity_hub_uuid(source: str, canonical_name: str) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#entity_hub#{canonical_name}',
    ).hex


def entity_hub_properties(
    source: str,
    canonical_name: str,
    description: str,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': entity_hub_uuid(source, canonical_name),
        'source': nodes.source_uuid(source),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }
