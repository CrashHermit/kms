"""KMS2-only Neo4j schema statements and initialization."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE CONSTRAINT source_uuid IF NOT EXISTS
    FOR (source:Source) REQUIRE source.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_block_uuid IF NOT EXISTS
    FOR (block:SourceBlock) REQUIRE block.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT visual_asset_uuid IF NOT EXISTS
    FOR (asset:VisualAsset) REQUIRE asset.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT triplet_uuid IF NOT EXISTS
    FOR (triplet:Triplet) REQUIRE triplet.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_uuid IF NOT EXISTS
    FOR (entity:Entity) REQUIRE entity.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT event_uuid IF NOT EXISTS
    FOR (event:Event) REQUIRE event.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT predicate_uuid IF NOT EXISTS
    FOR (predicate:Predicate) REQUIRE predicate.uuid IS UNIQUE
    """,
    """
    CREATE INDEX triplet_source_uuid IF NOT EXISTS
    FOR (triplet:Triplet) ON (triplet.source_uuid)
    """,
)


async def ensure_schema(
    session_factory: Callable[[], AbstractAsyncContextManager],
) -> None:
    """Create the KMS2 structural UUID constraints in order."""
    async with session_factory() as session:
        for statement in SCHEMA_STATEMENTS:
            await session.run(statement)
