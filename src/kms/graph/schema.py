"""
Schema bootstrap for the structural node layer, the statement overlay, and the
procedural layer.

Establishes the spine for the structural provenance layer (``:Node`` and its
``:Source`` root, see ``graph.nodes``), the ``:Statement`` overlay on top of it
(``graph.statements``), and the procedural layer (``:Procedure`` / ``:Act``,
``graph.procedures``): a uuid uniqueness constraint on each so ``MERGE`` on
uuid is safe and re-persisting a book never double-inserts, plus a ``source``
lookup index on ``:Node`` and ``:Statement`` so book-scoped lookups are
efficient. Idempotent DDL (``IF NOT EXISTS``), so ``ensure_schema`` is safe to
run on every startup.

The ``:Concept`` constraint is declared even though the concept layer is
currently dark (``graph.concepts``): the DDL is idempotent and cheap.
"""

from kms.graph import concepts, db, nodes, procedures, statements


def schema_statements() -> list[str]:
    """The idempotent DDL for the graph."""
    return [
        f'CREATE CONSTRAINT node_uuid IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT source_uuid IF NOT EXISTS '
        f'FOR (s:{nodes.SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT statement_uuid IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT procedure_uuid IF NOT EXISTS '
        f'FOR (p:{procedures.PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT act_uuid IF NOT EXISTS '
        f'FOR (a:{procedures.ACT_LABEL}) REQUIRE a.uuid IS UNIQUE',
        f'CREATE CONSTRAINT concept_uuid IF NOT EXISTS '
        f'FOR (c:{concepts.CONCEPT_LABEL}) REQUIRE c.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.source)',
        f'CREATE INDEX statement_source IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) ON (s.source)',
    ]


async def ensure_schema() -> None:
    """Create the constraints and index if absent. Idempotent — safe on
    every startup."""
    async with db.driver().session(database=db.database()) as session:
        for stmt in schema_statements():
            await session.run(stmt)
