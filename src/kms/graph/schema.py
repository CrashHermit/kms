"""
Schema bootstrap for the structural node layer and the entity overlay.

Establishes the spine for the structural provenance layer (``:Node`` and its ``:Source`` root,
see ``graph.nodes``), the ``:Entity`` overlay on top of it (``graph.entities``), and the procedural
layer (``:Procedure`` / ``:Event``, ``graph.procedures``): a uuid uniqueness constraint on each so
``MERGE`` on uuid is safe and re-persisting a book never double-inserts, plus a ``source`` lookup
index on ``:Node`` and ``:Entity`` so book-scoped lookups are efficient. Idempotent DDL
(``IF NOT EXISTS``), so ``ensure_schema`` is safe to run on every startup.

The reference-target **canonicals** (``graph.references``) carry the base ``:Entity`` label, so the
``:Entity`` uuid constraint already covers them — no separate canonical constraint is needed.

No index on the structural ``type`` or the entity ``type`` — both also carry per-type labels
(``:Math`` …, ``:Theorem`` …), so kind lookups are native label scans. And no vector index —
embeddings belong to the (undecided) semantic tiers above this layer, not to these vertices.
"""

from kms.graph.db import database, driver
from kms.graph.entities import ENTITY_LABEL
from kms.graph.nodes import NODE_LABEL, SOURCE_LABEL
from kms.graph.procedures import EVENT_LABEL, PROCEDURE_LABEL


def schema_statements() -> list[str]:
    """The idempotent DDL for the graph: uuid uniqueness keys on ``:Node``, ``:Source``, ``:Entity``
    (which also covers the reference canonicals — they are ``:Entity``), ``:Procedure`` and ``:Event``
    (the procedural layer), and a ``source`` lookup index on ``:Node`` and ``:Entity``."""
    return [
        f"CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.uuid IS UNIQUE",
        f"CREATE CONSTRAINT source_uuid IF NOT EXISTS "
        f"FOR (s:{SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE",
        f"CREATE CONSTRAINT entity_uuid IF NOT EXISTS "
        f"FOR (e:{ENTITY_LABEL}) REQUIRE e.uuid IS UNIQUE",
        f"CREATE CONSTRAINT procedure_uuid IF NOT EXISTS "
        f"FOR (p:{PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE",
        f"CREATE CONSTRAINT event_uuid IF NOT EXISTS "
        f"FOR (v:{EVENT_LABEL}) REQUIRE v.uuid IS UNIQUE",
        f"CREATE INDEX node_source IF NOT EXISTS FOR (n:{NODE_LABEL}) ON (n.source)",
        f"CREATE INDEX entity_source IF NOT EXISTS FOR (e:{ENTITY_LABEL}) ON (e.source)",
    ]


async def ensure_schema() -> None:
    """Create the structural-layer constraints and index if absent. Idempotent — safe on every
    startup before the tier writes."""
    async with driver().session(database=database()) as session:
        for statement in schema_statements():
            await session.run(statement)
