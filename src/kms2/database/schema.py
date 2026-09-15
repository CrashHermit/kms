"""KMS2-only Neo4j schema statements and initialization."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

SCHEMA_MIGRATION_STATEMENTS: tuple[str, ...] = (
    """
    MATCH (vertex:Entity)
    SET vertex:SourceEntity
    REMOVE vertex:Entity
    """,
    """
    MATCH (vertex:Event)
    SET vertex:SourceEvent
    REMOVE vertex:Event
    """,
    """
    MATCH (vertex:Predicate)
    SET vertex:SourcePredicate
    REMOVE vertex:Predicate
    """,
    'DROP CONSTRAINT entity_uuid IF EXISTS',
    'DROP CONSTRAINT event_uuid IF EXISTS',
    'DROP CONSTRAINT predicate_uuid IF EXISTS',
)

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
    CREATE CONSTRAINT instruction_uuid IF NOT EXISTS
    FOR (instruction:Instruction) REQUIRE instruction.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT statement_uuid IF NOT EXISTS
    FOR (statement:Statement) REQUIRE statement.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT procedure_uuid IF NOT EXISTS
    FOR (procedure:Procedure) REQUIRE procedure.uuid IS UNIQUE
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
    CREATE CONSTRAINT source_entity_uuid IF NOT EXISTS
    FOR (entity:SourceEntity) REQUIRE entity.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_event_uuid IF NOT EXISTS
    FOR (event:SourceEvent) REQUIRE event.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_predicate_uuid IF NOT EXISTS
    FOR (predicate:SourcePredicate) REQUIRE predicate.uuid IS UNIQUE
    """,
    """
    CREATE INDEX triplet_source_uuid IF NOT EXISTS
    FOR (triplet:Triplet) ON (triplet.source_uuid)
    """,
)


def vector_index_statements(embedding_dimension: int) -> tuple[str, ...]:
    """Return KMS2 cosine vector-index DDL for all persisted embeddings."""
    return tuple(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (node:{label}) ON (node.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {embedding_dimension},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        for index_name, label in (
            ('source_block_embedding', 'SourceBlock'),
            ('source_entity_embedding', 'SourceEntity'),
            ('source_event_embedding', 'SourceEvent'),
            ('source_predicate_embedding', 'SourcePredicate'),
        )
    )


VECTOR_INDEX_NAMES = (
    'source_block_embedding',
    'source_entity_embedding',
    'source_event_embedding',
    'source_predicate_embedding',
)
AWAIT_INDEX = 'CALL db.awaitIndex($index_name, $timeout_seconds)'


async def ensure_schema(
    session_factory: Callable[[], AbstractAsyncContextManager],
    *,
    embedding_dimension: int,
) -> None:
    """Migrate and create KMS2 structural and vector schema in order."""
    async with session_factory() as session:
        for statement in SCHEMA_MIGRATION_STATEMENTS:
            await session.run(statement)
        for statement in SCHEMA_STATEMENTS:
            await session.run(statement)
        for statement in vector_index_statements(embedding_dimension):
            await session.run(statement)
        for index_name in VECTOR_INDEX_NAMES:
            await session.run(
                AWAIT_INDEX,
                index_name=index_name,
                timeout_seconds=300,
            )
