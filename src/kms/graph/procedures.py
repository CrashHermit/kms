from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

PROCEDURE_LABEL = 'Procedure'
ACT_LABEL = 'Act'


def procedure_uuid(source: str, block: list[int], index: int) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#procedure#{nodes.block_key(block)}#{index}',
    ).hex


def act_uuid(
    source: str,
    statement_id: int,
    procedure_index: int,
    step_index: int,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#act#{statement_id}#{procedure_index}#{step_index}',
    ).hex


def procedure_properties(source: str, procedure: models.Procedure) -> dict:
    properties = {
        'uuid': procedure_uuid(source, procedure.block, procedure.index),
        'source': nodes.source_uuid(source),
        'index': procedure.index,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def act_properties(
    source: str,
    statement_id: int,
    procedure_index: int,
    step_index: int,
    text: str,
) -> dict:
    return {
        'uuid': act_uuid(source, statement_id, procedure_index, step_index),
        'source': nodes.source_uuid(source),
        'text': text,
        'index': step_index,
    }


def procedure_rows(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    return [procedure_properties(source, procedure) for procedure in procedures]


def act_rows(procedures: list[models.Procedure], source: str) -> list[dict]:
    return []


def procedure_member_pairs(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    return [
        {
            'node': nodes.node_uuid(source, member_id),
            'procedure': procedure_uuid(
                source, procedure.block, procedure.index
            ),
        }
        for procedure in procedures
        for member_id in procedure.members
    ]


def first_pairs(procedures: list[models.Procedure], source: str) -> list[dict]:
    return []


def then_pairs(procedures: list[models.Procedure], source: str) -> list[dict]:
    return []

