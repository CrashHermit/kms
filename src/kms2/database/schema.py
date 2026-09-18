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
    CREATE CONSTRAINT instruction_uuid IF NOT EXISTS
    FOR (instruction:Instruction) REQUIRE instruction.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT statement_uuid IF NOT EXISTS
    FOR (statement:SourceStatement) REQUIRE statement.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT procedure_uuid IF NOT EXISTS
    FOR (procedure:SourceProcedure) REQUIRE procedure.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT visual_asset_uuid IF NOT EXISTS
    FOR (asset:VisualAsset) REQUIRE asset.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT triplet_uuid IF NOT EXISTS
    FOR (triplet:SourceTriplet) REQUIRE triplet.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_fact_uuid IF NOT EXISTS
    FOR (fact:SourceFact) REQUIRE fact.uuid IS UNIQUE
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
    CREATE CONSTRAINT source_entity_hub_uuid IF NOT EXISTS
    FOR (hub:SourceEntityHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_event_hub_uuid IF NOT EXISTS
    FOR (hub:SourceEventHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_predicate_hub_uuid IF NOT EXISTS
    FOR (hub:SourcePredicateHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_statement_hub_uuid IF NOT EXISTS
    FOR (hub:SourceStatementHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_procedure_hub_uuid IF NOT EXISTS
    FOR (hub:SourceProcedureHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_triplet_hub_uuid IF NOT EXISTS
    FOR (hub:SourceTripletHub) REQUIRE hub.uuid IS UNIQUE
    """,
    """
    CREATE INDEX triplet_source_uuid IF NOT EXISTS
    FOR (triplet:SourceTriplet) ON (triplet.source_uuid)
    """,
    """
    CREATE INDEX source_fact_source_uuid IF NOT EXISTS
    FOR (fact:SourceFact) ON (fact.source_uuid)
    """,
    """
    CREATE INDEX source_triplet_hub_source_uuid IF NOT EXISTS
    FOR (hub:SourceTripletHub) ON (hub.source_uuid)
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
            ('source_statement_embedding', 'SourceStatement'),
            ('source_procedure_embedding', 'SourceProcedure'),
            ('source_entity_hub_embedding', 'SourceEntityHub'),
            ('source_event_hub_embedding', 'SourceEventHub'),
            ('source_predicate_hub_embedding', 'SourcePredicateHub'),
            ('source_statement_hub_embedding', 'SourceStatementHub'),
            ('source_procedure_hub_embedding', 'SourceProcedureHub'),
            ('source_triplet_hub_embedding', 'SourceTripletHub'),
        )
    )


VECTOR_INDEX_NAMES = (
    'source_block_embedding',
    'source_entity_embedding',
    'source_event_embedding',
    'source_predicate_embedding',
    'source_statement_embedding',
    'source_procedure_embedding',
    'source_entity_hub_embedding',
    'source_statement_hub_embedding',
    'source_procedure_hub_embedding',
    'source_event_hub_embedding',
    'source_predicate_hub_embedding',
    'source_triplet_hub_embedding',
)
AWAIT_INDEX = 'CALL db.awaitIndex($index_name, $timeout_seconds)'


async def ensure_schema(
    session_factory: Callable[[], AbstractAsyncContextManager],
    *,
    embedding_dimension: int,
) -> None:
    """Create KMS2 structural and vector schema in order."""
    async with session_factory() as session:
        for statement in SCHEMA_STATEMENTS:
            result = await session.run(statement)
            if result is not None:
                await result.consume()
        for statement in vector_index_statements(embedding_dimension):
            result = await session.run(statement)
            if result is not None:
                await result.consume()
        for index_name in VECTOR_INDEX_NAMES:
            result = await session.run(
                AWAIT_INDEX,
                index_name=index_name,
                timeout_seconds=300,
            )
            if result is not None:
                await result.consume()
