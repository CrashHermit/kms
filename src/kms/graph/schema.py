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
"""

from collections.abc import Callable

from kms.graph import (
    definitions,
    entities,
    entity_hubs,
    facts,
    instructions,
    nodes,
    predicate_hubs,
    predicates,
    procedures,
    statements,
    triplets,
)


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
        f'CREATE CONSTRAINT instruction_uuid IF NOT EXISTS '
        f'FOR (i:{instructions.INSTRUCTION_LABEL}) REQUIRE i.uuid IS UNIQUE',
        f'CREATE CONSTRAINT fact_uuid IF NOT EXISTS '
        f'FOR (f:{facts.FACT_LABEL}) REQUIRE f.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_uuid IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) REQUIRE e.uuid IS UNIQUE',
        f'CREATE CONSTRAINT predicate_uuid IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT triplet_uuid IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) REQUIRE t.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.source)',
        f'CREATE INDEX statement_source IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) ON (s.source)',
        f'CREATE INDEX fact_source IF NOT EXISTS '
        f'FOR (f:{facts.FACT_LABEL}) ON (f.source)',
        f'CREATE INDEX entity_source IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) ON (e.source)',
        f'CREATE INDEX predicate_source IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) ON (p.source)',
        f'CREATE INDEX triplet_source IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) ON (t.source)',
        # Vector indexes for semantic search over entity names +
        # descriptions and predicate text + descriptions. Dimensions
        # are set to a typical model default (1024) — Neo4j ignores
        # this for existing indexes and the embedder's actual output
        # dimension is what matters at write time.
        f'CREATE VECTOR INDEX entity_embedding IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) ON (e.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE VECTOR INDEX predicate_embedding IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) ON (p.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT entity_hub_uuid IF NOT EXISTS '
        f'FOR (h:{entity_hubs.ENTITY_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE CONSTRAINT predicate_hub_uuid IF NOT EXISTS '
        f'FOR (h:{predicate_hubs.PREDICATE_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE CONSTRAINT definition_uuid IF NOT EXISTS '
        f'FOR (d:{definitions.DEFINITION_LABEL}) '
        f'REQUIRE d.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX definition_embedding IF NOT EXISTS '
        f'FOR (d:{definitions.DEFINITION_LABEL}) '
        f'ON (d.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
    ]


async def ensure_schema(session_factory: Callable) -> None:
    """Create the constraints and index if absent. Idempotent — safe on
    every startup.

    Args:
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    async with session_factory() as session:
        for statement in schema_statements():
            await session.run(statement)
