"""
Graph representation of the procedural layer — the ``:Procedure`` containers
and their ``:Act`` step chains.

A statement's derivations are *procedures* — reified out of the statement as
real graph structure: one ``:Procedure`` per derivation, hung off its statement
via ``(:Statement)-[:HAS_PROCEDURE]->(:Procedure)``. ``:Act`` nodes for step
decomposition are declared but not yet written (the step decomposer is a
future pass).

Representation: a procedure carries the bare ``:Procedure`` label. An
``:Act`` carries only its verbatim ``text`` and its ordinal ``index``.

Identity: deterministic uuid5s, disjoint from node/statement uuids.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.statements import statement_uuid

PROCEDURE_LABEL = 'Procedure'
ACT_LABEL = 'Act'


def procedure_uuid(source: str, statement_id: int, index: int) -> str:
    """Stable, deterministic vertex key for a procedure."""
    return uuid5(
        NAMESPACE_URL, f'{source}#procedure#{statement_id}#{index}'
    ).hex


def act_uuid(
    source: str,
    statement_id: int,
    procedure_index: int,
    step_index: int,
) -> str:
    """Stable, deterministic vertex key for one procedure step."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#act#{statement_id}#{procedure_index}#{step_index}',
    ).hex


def procedure_properties(
    source: str, statement_id: int, procedure: models.Procedure
) -> dict:
    """The Neo4j property map for one procedure."""
    props = {
        'uuid': procedure_uuid(source, statement_id, procedure.index),
        'source': nodes.source_uuid(source),
        'index': procedure.index,
        'content': procedure.content,
    }
    return {key: value for key, value in props.items() if value is not None}


def act_properties(
    source: str,
    statement_id: int,
    procedure_index: int,
    step_index: int,
    text: str,
) -> dict:
    """The Neo4j property map for one procedure step."""
    return {
        'uuid': act_uuid(source, statement_id, procedure_index, step_index),
        'source': nodes.source_uuid(source),
        'text': text,
        'index': step_index,
    }


def procedure_rows(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """Every procedure's property map across the overlay, one flat list."""
    return [
        procedure_properties(source, statement.id, procedure)
        for statement in statements
        for procedure in statement.procedures
    ]


def act_rows(statements: list[models.Statement], source: str) -> list[dict]:
    """Every step's property map across the overlay. Currently empty —
    the step decomposer is a future pass."""
    return []


def has_procedure_pairs(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """The ``{statement, procedure}`` uuid pairs for ``:HAS_PROCEDURE``
    edges."""
    return [
        {
            'statement': statement_uuid(source, statement.id),
            'procedure': procedure_uuid(source, statement.id, procedure.index),
        }
        for statement in statements
        for procedure in statement.procedures
    ]


def first_pairs(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """The ``{procedure, act}`` uuid pairs for ``:FIRST`` edges. Empty
    until the step decomposer runs."""
    return []


def then_pairs(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:THEN`` chain. Empty
    until the step decomposer runs."""
    return []
