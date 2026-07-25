"""
Schema bootstrap for the structural node layer, the entity overlay, and the procedural layer.

Establishes the spine for the structural provenance layer (``:Node`` and its ``:Source`` root,
see ``graph.nodes``), the ``:Entity`` overlay on top of it (``graph.entities``), and the procedural
layer (``:Procedure`` / ``:Act``, ``graph.procedures``): a uuid uniqueness constraint on each so
``MERGE`` on uuid is safe and re-persisting a book never double-inserts, plus a ``source`` lookup
index on ``:Node`` and ``:Entity`` so book-scoped lookups are efficient. Idempotent DDL
(``IF NOT EXISTS``), so ``ensure_schema`` is safe to run on every startup.

The ``:Concept`` constraint is declared even though the concept layer is currently dark
(``graph.concepts``): the DDL is idempotent and cheap, and the hub identity scheme it protects is
already fixed, so conceptualization can start writing without a schema migration.

No index on the entity ``type``: it is an open induced property, so a lookup is a property scan
either way and an index over an unbounded vocabulary earns little. And no vector index — embeddings
belong to the (undecided) fusion tier above this layer, not to these vertices.
"""

from kms.graph import concepts, db, entities, nodes, procedures


def schema_statements() -> list[str]:
    """The idempotent DDL for the graph: uuid uniqueness keys on ``:Node``, ``:Source``,
    ``:Entity``, ``:Procedure`` and ``:Act`` (the procedural layer) and ``:Concept`` (dark, but
    declared), and a ``source`` lookup index on ``:Node`` and ``:Entity``."""
    return [
        f'CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:{nodes.NODE_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT source_uuid IF NOT EXISTS '
        f'FOR (s:{nodes.SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_uuid IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) REQUIRE e.uuid IS UNIQUE',
        f'CREATE CONSTRAINT procedure_uuid IF NOT EXISTS '
        f'FOR (p:{procedures.PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT act_uuid IF NOT EXISTS '
        f'FOR (a:{procedures.ACT_LABEL}) REQUIRE a.uuid IS UNIQUE',
        f'CREATE CONSTRAINT concept_uuid IF NOT EXISTS '
        f'FOR (c:{concepts.CONCEPT_LABEL}) REQUIRE c.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS FOR (n:{nodes.NODE_LABEL}) ON (n.source)',
        f'CREATE INDEX entity_source IF NOT EXISTS FOR (e:{entities.ENTITY_LABEL}) ON (e.source)',
    ]


async def ensure_schema() -> None:
    """Create the structural-layer constraints and index if absent. Idempotent — safe on every
    startup before the tier writes."""
    async with db.driver().session(database=db.database()) as session:
        for statement in schema_statements():
            await session.run(statement)
