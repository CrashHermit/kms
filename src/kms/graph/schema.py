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

The structural ``:Node`` layer needs no ``type`` index: its type comes from the closed
``models.NodeType`` and is also a per-type label, so a kind lookup is a native label scan. The
semantic layers are the opposite case — since the type vocabulary opened up
(``docs/GENERALIZATION.md``), an ``:Entity`` / ``:Procedure`` carries its type only as a property, so
each gets a ``type`` index to keep "all the theorems" as cheap as the label scan it replaced. The
concept layer gets a ``name`` index for the same reason: concept lookup is by name, and the semantic
dedup tier will scan it. No vector index — embeddings belong to the (undecided) fusion tier above this
layer, not to these vertices.
"""

from kms.graph.concepts import CONCEPT_LABEL
from kms.graph.db import database, driver
from kms.graph.entities import ENTITY_LABEL
from kms.graph.nodes import NODE_LABEL, SOURCE_LABEL
from kms.graph.procedures import EVENT_LABEL, PROCEDURE_LABEL


def schema_statements() -> list[str]:
    """The idempotent DDL for the graph: uuid uniqueness keys on ``:Node``, ``:Source``, ``:Entity``
    (which also covers the reference canonicals — they are ``:Entity``), ``:Procedure`` and ``:Event``
    (the procedural layer) and ``:Concept`` (the concept layer); a ``source`` lookup index on
    ``:Node`` and ``:Entity``; and — standing in for the per-type labels the semantic layers no longer
    mint — a ``type`` index on ``:Entity`` and ``:Procedure`` plus a ``name`` index on ``:Concept``."""
    return [
        f'CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT source_uuid IF NOT EXISTS '
        f'FOR (s:{SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_uuid IF NOT EXISTS '
        f'FOR (e:{ENTITY_LABEL}) REQUIRE e.uuid IS UNIQUE',
        f'CREATE CONSTRAINT procedure_uuid IF NOT EXISTS '
        f'FOR (p:{PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT event_uuid IF NOT EXISTS '
        f'FOR (v:{EVENT_LABEL}) REQUIRE v.uuid IS UNIQUE',
        f'CREATE CONSTRAINT concept_uuid IF NOT EXISTS '
        f'FOR (c:{CONCEPT_LABEL}) REQUIRE c.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS FOR (n:{NODE_LABEL}) ON (n.source)',
        f'CREATE INDEX entity_source IF NOT EXISTS FOR (e:{ENTITY_LABEL}) ON (e.source)',
        f'CREATE INDEX entity_type IF NOT EXISTS FOR (e:{ENTITY_LABEL}) ON (e.type)',
        f'CREATE INDEX procedure_type IF NOT EXISTS '
        f'FOR (p:{PROCEDURE_LABEL}) ON (p.type)',
        f'CREATE INDEX concept_name IF NOT EXISTS FOR (c:{CONCEPT_LABEL}) ON (c.name)',
    ]


async def ensure_schema() -> None:
    """Create the structural-layer constraints and index if absent. Idempotent — safe on every
    startup before the tier writes."""
    async with driver().session(database=database()) as session:
        for statement in schema_statements():
            await session.run(statement)
