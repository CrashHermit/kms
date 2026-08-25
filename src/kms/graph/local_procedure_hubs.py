from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

LOCAL_PROCEDURE_HUB_LABEL = 'LocalProcedureHub'
GLOBAL_PROCEDURE_HUB_LABEL = 'GlobalProcedureHub'


def hub_label(tier: str = 'local') -> str:
    if tier == 'local':
        return LOCAL_PROCEDURE_HUB_LABEL
    if tier == 'global':
        return GLOBAL_PROCEDURE_HUB_LABEL
    raise ValueError(f'unknown procedure hub tier: {tier}')


def base_label() -> str:
    return 'Procedure'


def hub_uuid(source: str, members: list[str]) -> str:
    if not members:
        raise ValueError('procedure hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'{source}#procedure_hub#{identity}').hex


def global_hub_uuid(members: list[str]) -> str:
    if not members:
        raise ValueError('global procedure hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'global#procedure_hub#{identity}').hex


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
