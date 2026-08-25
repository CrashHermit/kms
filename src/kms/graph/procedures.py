"""Row and edge builders for the procedure layer."""

from kms.core import identity, models
from kms.graph import nodes

PROCEDURE_LABEL = 'Procedure'


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
    return identity.procedure_uuid(
        source, block, index, statement_uuid_value=statement_uuid
    )


def _procedure_id(source: str, procedure: models.Procedure) -> str:
    """Returns and verifies the persisted UUID for a procedure model."""
    expected = procedure_uuid(
        source,
        procedure.block,
        procedure.index,
        statement_uuid=procedure.statement_uuid,
    )
    if procedure.uuid is None:
        raise ValueError('procedure is missing its assigned uuid')
    if procedure.uuid != expected:
        raise ValueError(
            f'procedure uuid {procedure.uuid!r} does not match expected '
            f'{expected!r}'
        )
    return procedure.uuid


def procedure_properties(source: str, procedure: models.Procedure) -> dict:
    """Builds the property dict used to persist a procedure."""
    properties = {
        'uuid': _procedure_id(source, procedure),
        'source': nodes.source_uuid(source),
        'index': procedure.index,
        'kind': procedure.kind.value,
        'procedure': procedure.procedure,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def procedure_enrichment_properties(
    procedure_uuid_value: str,
    procedure: str,
    embedding: list[float],
) -> dict:
    """Builds compiled content and its embedding for one Procedure."""
    return {
        'uuid': procedure_uuid_value,
        'procedure': procedure,
        'embedding': embedding,
    }

def procedure_rows(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    """Builds the row dicts for all procedures in a source."""
    return [procedure_properties(source, procedure) for procedure in procedures]


def procedure_member_pairs(
    procedures: list[models.Procedure],
    source: str,
    doc_nodes: list[models.SourceNode],
) -> list[dict]:
    """Builds procedure→member node edge pairs."""
    pairs: list[dict] = []
    for procedure in procedures:
        procedure_id = _procedure_id(source, procedure)
        for member_position in procedure.member_positions:
            if not 0 <= member_position < len(doc_nodes):
                raise ValueError(
                    f'procedure member position {member_position} is outside '
                    f'the node stream'
                )
            node = doc_nodes[member_position]
            pairs.append(
                {
                    'node': nodes.node_uuid(source, node),
                    'procedure': procedure_id,
                }
            )
    return pairs
