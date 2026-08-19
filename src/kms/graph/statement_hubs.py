from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

STATEMENT_HUB_LABEL = 'StatementHub'
META_STATEMENT_HUB_LABEL = 'MetaStatementHub'


def hub_label(tier: str = 'source') -> str:
    if tier == 'source':
        return STATEMENT_HUB_LABEL
    if tier == 'meta':
        return META_STATEMENT_HUB_LABEL
    raise ValueError(f'unknown statement hub tier: {tier}')


def base_label() -> str:
    return 'Statement'


def hub_uuid(source: str, members: list[str]) -> str:
    if not members:
        raise ValueError('statement hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'{source}#statement_hub#{identity}').hex


def meta_hub_uuid(members: list[str]) -> str:
    if not members:
        raise ValueError('meta statement hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'meta#statement_hub#{identity}').hex


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


def meta_hub_properties(
    canonical_name: str,
    description: str,
    embedding: list[float],
    members: list[str],
) -> dict:
    return {
        'uuid': meta_hub_uuid(members),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }
