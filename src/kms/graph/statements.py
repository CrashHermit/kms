"""Row and edge builders for the statement layer."""

from kms.core import identity, models
from kms.graph import nodes

STATEMENT_LABEL = 'Statement'


def statement_uuid(source: str, block: list[int]) -> str:
    """Compatibility export for the canonical identity function."""
    return identity.statement_uuid(source, block)


def _statement_id(statement: models.Statement, source: str) -> str:
    """Returns and verifies the assigned model identity."""
    expected = identity.statement_uuid(source, statement.block)
    if statement.uuid is None:
        raise ValueError('statement is missing its assigned uuid')
    if statement.uuid != expected:
        raise ValueError(
            f'statement uuid {statement.uuid!r} does not match expected '
            f'{expected!r}'
        )
    return statement.uuid


def statement_properties(statement: models.Statement, source: str) -> dict:
    """Builds the property dict used to persist a statement."""
    properties = {
        'uuid': _statement_id(statement, source),
        'source': nodes.source_uuid(source),
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def statement_enrichment_properties(
    statement_uuid_value: str,
    description: str,
    embedding: list[float],
) -> dict:
    """Builds derived semantic properties for one Statement."""
    return {
        'uuid': statement_uuid_value,
        'description': description,
        'embedding': embedding,
    }


def statement_member_pairs(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """Builds statement→member node edge pairs."""
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'statement': _statement_id(statement, source),
        }
        for statement in statements
        for node_id in statement.members
    ]


def has_procedure_pairs(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    source: str,
) -> list[dict]:
    """Builds statement→procedure edges for blocks that are both.

    Some blocks are found both as declarative statements and as
    procedures; this links the statement vertex to the procedure so
    the dual identity is explicit in the graph.
    """
    pairs: list[dict] = []
    proc_by_block: dict[tuple, models.Procedure] = {}
    for proc in procedures:
        proc_by_block[tuple(proc.block)] = proc

    for statement in statements:
        key = tuple(statement.block)
        if key in proc_by_block:
            pairs.append(
                {
                    'statement': _statement_id(statement, source),
                    'procedure': identity.procedure_uuid(
                        source,
                        statement.block,
                        proc_by_block[key].index,
                        statement_uuid_value=proc_by_block[key].statement_uuid,
                    ),
                }
            )
    return pairs
