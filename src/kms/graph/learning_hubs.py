from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

STATEMENT_HUB_LABEL = 'StatementHub'
PROCEDURE_HUB_LABEL = 'ProcedureHub'
META_STATEMENT_HUB_LABEL = 'MetaStatementHub'
META_PROCEDURE_HUB_LABEL = 'MetaProcedureHub'

_LABELS = {
    'statement': STATEMENT_HUB_LABEL,
    'procedure': PROCEDURE_HUB_LABEL,
}
_META_LABELS = {
    'statement': META_STATEMENT_HUB_LABEL,
    'procedure': META_PROCEDURE_HUB_LABEL,
}
_BASE_LABELS = {
    'statement': 'Statement',
    'procedure': 'Procedure',
}


def hub_label(kind: str, tier: str = 'source') -> str:
    labels = _LABELS if tier == 'source' else _META_LABELS
    if tier not in {'source', 'meta'}:
        raise ValueError(f'unknown learning hub tier: {tier}')
    try:
        return labels[kind]
    except KeyError as error:
        raise ValueError(f'unknown learning hub kind: {kind}') from error


def base_label(kind: str) -> str:
    try:
        return _BASE_LABELS[kind]
    except KeyError as error:
        raise ValueError(f'unknown learning hub kind: {kind}') from error


def hub_uuid(kind: str, source: str, members: list[str]) -> str:
    if not members:
        raise ValueError('learning hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(
        NAMESPACE_URL,
        f'{source}#{kind}_hub#{identity}',
    ).hex


def meta_hub_uuid(kind: str, members: list[str]) -> str:
    if not members:
        raise ValueError('meta learning hubs require at least one member')
    identity = '#'.join(sorted(members))
    return uuid5(NAMESPACE_URL, f'meta#{kind}_hub#{identity}').hex


def hub_properties(
    kind: str,
    source: str,
    canonical_name: str,
    description: str,
    embedding: list[float],
    members: list[str],
) -> dict:
    return {
        'uuid': hub_uuid(kind, source, members),
        'source': nodes.source_uuid(source),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }


def meta_hub_properties(
    kind: str,
    canonical_name: str,
    description: str,
    embedding: list[float],
    members: list[str],
) -> dict:
    return {
        'uuid': meta_hub_uuid(kind, members),
        'canonical_name': canonical_name,
        'description': description,
        'embedding': embedding,
    }
