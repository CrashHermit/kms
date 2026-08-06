"""
Graph representation of the procedural layer — the ``:Procedure`` hubs and
their ``:Act`` step chains.

A derivation is its own hub, independent of any statement: one ``:Procedure``
per derivation, pointing at the raw blocks that are its portion via
``(:Node)-[:MEMBER_OF]->(:Procedure)``. There is deliberately NO edge from a
``:Statement`` to its derivation — the statement↔procedure relationship is
parked for the semantic tier. ``:Act`` nodes for step decomposition are
declared but not yet written (the step decomposer is a future pass).

Representation: a procedure carries the bare ``:Procedure`` label — a hub
with no text of its own; its members are its body. An ``:Act`` carries only
its verbatim ``text`` and its ordinal ``index``.

Identity: deterministic uuid5s over the procedure's WHOLE block (its member
node ids, frozen at creation) plus its index, disjoint from node/statement
uuids.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

PROCEDURE_LABEL = 'Procedure'
ACT_LABEL = 'Act'


def procedure_uuid(source: str, block: list[int], index: int) -> str:
    """Stable, deterministic vertex key for a procedure.

    Args:
        source: The stable book identity.
        block: The PCF block's member node ids, in document order.
        index: The procedure's position within its block.

    Returns:
        The procedure's hex uuid, disjoint from every node/statement uuid.
    """
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
    """Stable, deterministic vertex key for one procedure step."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#act#{statement_id}#{procedure_index}#{step_index}',
    ).hex


def procedure_properties(source: str, procedure: models.Procedure) -> dict:
    """The Neo4j property map for one procedure."""
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
    """The Neo4j property map for one procedure step."""
    return {
        'uuid': act_uuid(source, statement_id, procedure_index, step_index),
        'source': nodes.source_uuid(source),
        'text': text,
        'index': step_index,
    }


def procedure_rows(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    """Every procedure's property map across the overlay, one flat list."""
    return [procedure_properties(source, procedure) for procedure in procedures]


def act_rows(procedures: list[models.Procedure], source: str) -> list[dict]:
    """Every step's property map across the overlay. Currently empty —
    the step decomposer is a future pass."""
    return []


def procedure_member_pairs(
    procedures: list[models.Procedure], source: str
) -> list[dict]:
    """The ``{node, procedure}`` uuid pairs for ``:MEMBER_OF`` edges.

    A procedure's members are its portion of its block — the raw nodes that
    are its body. Each member gets one edge to the procedure.

    Args:
        procedures: The procedure overlay.
        source: The stable book identity.

    Returns:
        One ``{node, procedure}`` per member node.
    """
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
    """The ``{procedure, act}`` uuid pairs for ``:FIRST`` edges. Empty
    until the step decomposer runs."""
    return []


def then_pairs(procedures: list[models.Procedure], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:THEN`` chain. Empty
    until the step decomposer runs."""
    return []
