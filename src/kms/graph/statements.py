from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.procedures import procedure_uuid

STATEMENT_LABEL = 'Statement'


def statement_uuid(source: str, block: list[int]) -> str:
    return uuid5(
        NAMESPACE_URL, f'{source}#statement#{nodes.block_key(block)}'
    ).hex


def statement_properties(statement: models.Statement, source: str) -> dict:
    properties = {
        'uuid': statement_uuid(source, statement.block),
        'source': nodes.source_uuid(source),
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def statement_member_pairs(
    statements: list[models.Statement], source: str
) -> list[dict]:
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'statement': statement_uuid(source, statement.block),
        }
        for statement in statements
        for node_id in statement.members
    ]


def has_procedure_pairs(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    source: str,
) -> list[dict]:
    pairs: list[dict] = []
    proc_by_block: dict[tuple, models.Procedure] = {}
    for proc in procedures:
        proc_by_block[tuple(proc.block)] = proc

    for statement in statements:
        key = tuple(statement.block)
        if key in proc_by_block:
            pairs.append(
                {
                    'statement': statement_uuid(source, statement.block),
                    'procedure': procedure_uuid(
                        source, statement.block, proc_by_block[key].index
                    ),
                }
            )
    return pairs
