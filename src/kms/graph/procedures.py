from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

PROCEDURE_LABEL = 'Procedure'
STEP_LABEL = 'Step'


def procedure_uuid(
    source: str,
    block: list[int],
    index: int,
    statement_uuid: str | None = None,
) -> str:
    key = f'{source}#procedure#{nodes.block_key(block)}#{index}'
    if statement_uuid is not None:
        key = f'{key}#{statement_uuid}'
    return uuid5(NAMESPACE_URL, key).hex


def step_uuid(source: str, procedure_uuid: str, step_index: int) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#step#{procedure_uuid}#{step_index}',
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


def step_properties(
    source: str,
    procedure_uuid: str,
    step_index: int,
    text: str,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': step_uuid(source, procedure_uuid, step_index),
        'source': nodes.source_uuid(source),
        'text': text,
        'index': step_index,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def procedure_rows(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    return [procedure_properties(source, procedure) for procedure in procedures]


def step_rows(procedures: list[models.Procedure], source: str) -> list[dict]:
    rows: list[dict] = []
    for procedure in procedures:
        if not procedure.steps:
            continue
        procedure_id = procedure_uuid(source, procedure.block, procedure.index)
        for step in procedure.steps:
            rows.append(
                step_properties(source, procedure_id, step.index, step.text)
            )
    return rows


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
    pairs: list[dict] = []
    for procedure in procedures:
        if not procedure.steps:
            continue
        procedure_id = procedure_uuid(source, procedure.block, procedure.index)
        pairs.append(
            {
                'procedure': procedure_id,
                'step': step_uuid(
                    source, procedure_id, procedure.steps[0].index
                ),
            }
        )
    return pairs


def then_pairs(procedures: list[models.Procedure], source: str) -> list[dict]:
    pairs: list[dict] = []
    for procedure in procedures:
        if len(procedure.steps) < 2:
            continue
        procedure_id = procedure_uuid(source, procedure.block, procedure.index)
        for current, following in zip(
            procedure.steps, procedure.steps[1:], strict=False
        ):
            pairs.append(
                {
                    'from': step_uuid(source, procedure_id, current.index),
                    'to': step_uuid(source, procedure_id, following.index),
                }
            )
    return pairs
