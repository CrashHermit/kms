"""Row and edge builders for the procedure and step layers."""

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
    """Returns the deterministic uuid for a procedure.

    Args:
        source: The source key.
        block: The member node block.
        index: The procedure's index within the source.
        statement_uuid: Optional statement uuid to disambiguate a
            procedure that also exists as a statement.
    """
    key = f'{source}#procedure#{nodes.block_key(block)}#{index}'
    if statement_uuid is not None:
        key = f'{key}#{statement_uuid}'
    return uuid5(NAMESPACE_URL, key).hex


def step_uuid(source: str, procedure_uuid: str, step_index: int) -> str:
    """Returns the deterministic uuid for one step within a procedure."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#step#{procedure_uuid}#{step_index}',
    ).hex


def _procedure_id(source: str, procedure: models.Procedure) -> str:
    """Returns the persisted UUID for a procedure model."""
    return procedure_uuid(
        source,
        procedure.block,
        procedure.index,
        statement_uuid=procedure.statement_uuid,
    )


def procedure_properties(source: str, procedure: models.Procedure) -> dict:
    """Builds the property dict used to persist a procedure."""
    properties = {
        'uuid': _procedure_id(source, procedure),
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
    """Builds the property dict used to persist one step.

    Args:
        source: The source key.
        procedure_uuid: The owning procedure's uuid.
        step_index: The step's index within the procedure.
        text: The step text.
        embedding: Optional vector embedding for the step text.
    """
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
    """Builds the row dicts for all procedures in a source."""
    return [procedure_properties(source, procedure) for procedure in procedures]


def step_rows(procedures: list[models.Procedure], source: str) -> list[dict]:
    """Builds the row dicts for every step of every procedure."""
    rows: list[dict] = []
    for procedure in procedures:
        if not procedure.steps:
            continue
        procedure_id = _procedure_id(source, procedure)
        for step in procedure.steps:
            rows.append(
                step_properties(source, procedure_id, step.index, step.text)
            )
    return rows


def procedure_member_pairs(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    """Builds procedure→member node edge pairs."""
    return [
        {
            'node': nodes.node_uuid(source, member_id),
            'procedure': _procedure_id(source, procedure),
        }
        for procedure in procedures
        for member_id in procedure.members
    ]


def first_pairs(procedures: list[models.Procedure], source: str) -> list[dict]:
    """Builds procedure→first-step edge pairs."""
    pairs: list[dict] = []
    for procedure in procedures:
        if not procedure.steps:
            continue
        procedure_id = _procedure_id(source, procedure)
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
    """Builds step→step edge pairs for consecutive procedure steps."""
    pairs: list[dict] = []
    for procedure in procedures:
        if len(procedure.steps) < 2:
            continue
        procedure_id = _procedure_id(source, procedure)
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
