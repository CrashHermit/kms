from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

LOCAL_STATEMENT_HUB_LABEL = 'LocalStatementHub'
GLOBAL_STATEMENT_HUB_LABEL = 'GlobalStatementHub'


def hub_label(tier: str = 'local') -> str:
    if tier == 'local':
        return LOCAL_STATEMENT_HUB_LABEL
    if tier == 'global':
        return GLOBAL_STATEMENT_HUB_LABEL
    raise ValueError(f'unknown statement hub tier: {tier}')


def base_label() -> str:
    return 'Statement'


def hub_uuid(source: str, members: list[str]) -> str:
    if not members:
        raise ValueError('statement hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'{source}#statement_hub#{identity}').hex


def global_hub_uuid(members: list[str]) -> str:
    if not members:
        raise ValueError('global statement hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'global#statement_hub#{identity}').hex


def hub_properties(
    source: str,
    canonical_name: str,
    description: str,
    embedding: list[float],
    members: list[str],
) -> dict:
    return {
        'uuid': hub_uuid(source, members),
        'source': nodes.source_uuid(source),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }


def global_hub_properties(
    canonical_name: str,
    description: str,
    embedding: list[float],
    members: list[str],
) -> dict:
    return {
        'uuid': global_hub_uuid(members),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }
