"""
Graph representation of the statement overlay — the semantic tier over the
node stream.

The PCF groups related content into compound spans; the role typer diagnoses
each group and creates ``models.Statement``s with optional attached Procedures.
This module maps one onto its Neo4j form, mirroring ``graph.nodes`` for the
structural layer: pure mapping, free of the neo4j driver (the driver lives in
``graph.db``, the writes in ``graph.writer``).

Representation: every statement carries the bare ``:Statement`` label — a
HUB with no text of its own: its members' raw blocks carry the text, and a
statement points at them via ``(:Node)-[:MEMBER_OF]->(:Statement)`` — one edge
per member of its portion. Reaching a statement starts from its members, never
from a chain walk.

Identity: the stable vertex key is a DETERMINISTIC uuid5 over the statement's
WHOLE block — its member node ids, frozen at creation (see ``models.Statement``)
— so re-persisting a book MERGEs onto the same vertices and overlapping spans
never collide. The ``statement#`` segment keeps these uuids DISJOINT from node
uuids, as the ``procedure#`` segment does for procedures.

That disjointness is load-bearing. A statement and its member nodes name the
same PLACE in the document but are two different things — the nodes are the
verbatim blocks, the statement is the whole group — and they are vertices in
two tiers. Deriving the statement's uuid from a single member's would collapse
them onto one key, which was only survivable while the two were literally one
vertex: the role typer used to put the statement INTO the node stream, and the
persister wrote it as a fused ``:Node:Statement`` carrying the group's text in
place of the block's own. With the tiers properly separated, a shared key would
instead leave two vertices answering to one uuid and make every label-scoped
``MATCH (a {uuid: …})`` ambiguous.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

STATEMENT_LABEL = 'Statement'


def statement_uuid(source: str, block: list[int]) -> str:
    """Stable, deterministic vertex key for a statement.

    The identity is the block's WHOLE id set, not a single anchor node: a
    statement and its procedure share the block, and the ``statement#``
    segment keeps them apart. Frozen at hub creation, so partitioning never
    moves it.

    Args:
        source: The stable book identity.
        block: The PCF block's member node ids, in document order.

    Returns:
        The statement's hex uuid, disjoint from every node uuid.
    """
    return uuid5(
        NAMESPACE_URL, f'{source}#statement#{nodes.block_key(block)}'
    ).hex


def statement_properties(statement: models.Statement, source: str) -> dict:
    """The Neo4j property map for one statement.

    Args:
        statement: The statement to map.
        source: The stable book identity.

    Returns:
        The property map, with None values omitted.
    """
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
    """The ``{node, statement}`` uuid pairs for ``:MEMBER_OF`` edges.

    Every member node of a statement's portion is a member of it — the raw
    blocks that are its body — so each member gets one edge to the statement.
    All members are linked explicitly: the chain never skips absorbed nodes,
    so the relationship has to be real graph structure.

    Args:
        statements: The statement overlay.
        source: The stable book identity.

    Returns:
        One ``{node, statement}`` per member node.
    """
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'statement': statement_uuid(source, statement.block),
        }
        for statement in statements
        for node_id in statement.members
    ]
