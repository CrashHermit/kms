"""
Schema bootstrap for the structural node layer.

Establishes the spine for the one node kind we've committed to (``:Node``, see
``graph.nodes``): a uuid uniqueness constraint so ``MERGE`` on uuid is safe and re-persisting a
book never double-inserts, and a lookup index on the structural ``type``. Idempotent DDL
(``IF NOT EXISTS``), so ``ensure_schema`` is safe to run on every startup.

No vector index here — embeddings belong to the (undecided) semantic tiers above this layer,
not to the structural provenance nodes.
"""

from kms.graph.db import database, driver
from kms.graph.nodes import NODE_LABEL


def schema_statements() -> list[str]:
    """The idempotent DDL for the structural node layer: a uuid uniqueness key and a ``type``
    lookup index on ``:Node``."""
    return [
        f"CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.uuid IS UNIQUE",
        f"CREATE INDEX node_type IF NOT EXISTS FOR (n:{NODE_LABEL}) ON (n.type)",
    ]


async def ensure_schema() -> None:
    """Create the structural-node constraint and index if absent. Idempotent — safe on every
    startup before the tier writes."""
    async with driver().session(database=database()) as session:
        for statement in schema_statements():
            await session.run(statement)
