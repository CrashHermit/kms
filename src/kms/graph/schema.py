"""
Schema bootstrap for the structural node layer.

Establishes the spine for the structural provenance layer (``:Node`` and its ``:Source`` root,
see ``graph.nodes``): a uuid uniqueness constraint on each so ``MERGE`` on uuid is safe and
re-persisting a book never double-inserts, plus an index on ``:Node(source)`` so book-scoped
lookups are efficient. Idempotent DDL (``IF NOT EXISTS``), so ``ensure_schema`` is safe to run on
every startup.

No index on the structural ``type`` — nodes also carry a per-type label (``:Math``, …), so kind
lookups are native label scans. And no vector index — embeddings belong to the (undecided)
semantic tiers above this layer, not to the structural provenance nodes.
"""

from kms.graph.db import database, driver
from kms.graph.nodes import NODE_LABEL, SOURCE_LABEL


def schema_statements() -> list[str]:
    """The idempotent DDL for the structural layer: uuid uniqueness keys on ``:Node`` and
    ``:Source``, and a ``source`` lookup index on ``:Node``."""
    return [
        f"CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.uuid IS UNIQUE",
        f"CREATE CONSTRAINT source_uuid IF NOT EXISTS "
        f"FOR (s:{SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE",
        f"CREATE INDEX node_source IF NOT EXISTS FOR (n:{NODE_LABEL}) ON (n.source)",
    ]


async def ensure_schema() -> None:
    """Create the structural-node constraint if absent. Idempotent — safe on every startup
    before the tier writes."""
    async with driver().session(database=database()) as session:
        for statement in schema_statements():
            await session.run(statement)
