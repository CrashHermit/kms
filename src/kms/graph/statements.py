"""
Graph representation of the statement overlay — the semantic tier over the
node stream.

The PCF groups related content into compound spans; the role typer diagnoses
each group and creates StatementNodes with optional attached Procedures. This
module maps a ``core.models.StatementNode`` onto its Neo4j form, mirroring
``graph.nodes`` for the structural layer: pure mapping, free of the neo4j
driver (the driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: every statement carries the bare ``:Statement`` label. Its
``content`` is a single string written by the statement extractor.

Identity: the stable vertex key is a DETERMINISTIC uuid5 over
``(source, id)`` — the id is the statement's document-order position — so
re-persisting a book MERGEs onto the same vertices. The ``statement#``
segment keeps these uuids disjoint from node uuids.
"""

from kms.core import models
from kms.graph import nodes

STATEMENT_LABEL = 'Statement'


def statement_uuid(source: str, first_node_id: int) -> str:
    """Stable vertex key for a statement — same as ``node_uuid`` for
    the statement's first member node, so ``persist_statements`` MERGEs
    onto the existing ``:Node:Statement`` vertex."""
    return nodes.node_uuid(source, first_node_id)


def statement_properties(
    statement: models.StatementNode, source: str
) -> dict:
    """The Neo4j property map for one statement: matches the uuid of
    the existing ``:Node:Statement`` vertex and sets ``content``."""
    props = {
        'uuid': statement_uuid(source, statement.first_node_id()),
        'content': statement.content,
    }
    return {
        key: value for key, value in props.items() if value is not None
    }
