"""
Graph representation of the statement overlay — the semantic tier over the
node stream.

The PCF groups related content into compound spans; the role typer diagnoses
each group and creates ``models.Statement``s with optional attached Procedures.
This module maps one onto its Neo4j form, mirroring ``graph.nodes`` for the
structural layer: pure mapping, free of the neo4j driver (the driver lives in
``graph.db``, the writes in ``graph.writer``).

Representation: every statement carries the bare ``:Statement`` label. Its
``content`` is a single string written by the statement extractor.

Identity: the stable vertex key is a DETERMINISTIC uuid5 over
``(source, id)`` — the id is the statement's document-order position, i.e. its
first member node's id — so re-persisting a book MERGEs onto the same vertices.
The ``statement#`` segment keeps these uuids DISJOINT from node uuids, as the
``procedure#`` segment does for procedures.

That disjointness is load-bearing. A statement and its first member node name
the same PLACE in the document but are two different things — the node is that
one verbatim block, the statement is the whole group — and they are two
vertices in two tiers. Deriving the statement's uuid from the node's (which
this module used to do) collapsed them onto one key, which was only survivable
while the two were literally one vertex: the role typer put the statement INTO
the node stream, so the persister wrote it as a fused ``:Node:Statement``
carrying the group's text in place of the block's own. With the tiers properly
separated, a shared key would instead leave two vertices answering to one uuid
and make the chain's ``MATCH (a {uuid: …})`` ambiguous.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

STATEMENT_LABEL = 'Statement'


def statement_uuid(source: str, statement_id: int) -> str:
    """Stable, deterministic vertex key for a statement.

    Args:
        source: The stable book identity.
        statement_id: The statement's document-order position (its first
            member node's id).

    Returns:
        The statement's hex uuid, disjoint from every node uuid.
    """
    return uuid5(NAMESPACE_URL, f'{source}#statement#{statement_id}').hex


def statement_properties(statement: models.Statement, source: str) -> dict:
    """The Neo4j property map for one statement.

    Args:
        statement: The statement to map.
        source: The stable book identity.

    Returns:
        The property map, with None values omitted.
    """
    props = {
        'uuid': statement_uuid(source, statement.id),
        'source': nodes.source_uuid(source),
        'content': statement.content,
    }
    return {key: value for key, value in props.items() if value is not None}
