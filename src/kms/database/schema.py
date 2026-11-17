"""KMS2-only Neo4j schema statements and initialization."""

from collections.abc import Callable

from neo4j import AsyncSession

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE CONSTRAINT source_uuid IF NOT EXISTS
    FOR (source:Source) REQUIRE source.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_unit_uuid IF NOT EXISTS
    FOR (unit:SourceUnit) REQUIRE unit.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_block_uuid IF NOT EXISTS
    FOR (block:SourceBlock) REQUIRE block.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_uuid IF NOT EXISTS
    FOR (user:User) REQUIRE user.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT proposition_uuid IF NOT EXISTS
    FOR (proposition:Proposition) REQUIRE proposition.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT argument_uuid IF NOT EXISTS
    FOR (argument:Argument) REQUIRE argument.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learning_session_uuid IF NOT EXISTS
    FOR (session:LearningSession) REQUIRE session.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learning_session_step_uuid IF NOT EXISTS
    FOR (step:LearningSessionStep) REQUIRE step.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT task_response_uuid IF NOT EXISTS
    FOR (response:TaskResponse) REQUIRE response.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT task_assessment_uuid IF NOT EXISTS
    FOR (assessment:TaskAssessment) REQUIRE assessment.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learning_task_uuid IF NOT EXISTS
    FOR (task:Task) REQUIRE task.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learning_solution_uuid IF NOT EXISTS
    FOR (solution:Solution) REQUIRE solution.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learning_memory_uuid IF NOT EXISTS
    FOR (memory:LearningMemory) REQUIRE memory.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT memory_review_uuid IF NOT EXISTS
    FOR (review:MemoryReview) REQUIRE review.uuid IS UNIQUE
    """,
    """
    CREATE RANGE INDEX learning_memory_due IF NOT EXISTS
    FOR (memory:LearningMemory) ON (memory.due_at)
    """,
)


def vector_index_statements(embedding_dimension: int) -> tuple[str, ...]:
    """Return DDL for embeddings queried through Neo4j vector indexes."""
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
            ('source_unit_embedding', 'SourceUnit'),
        )
    )


VECTOR_INDEX_NAMES = (
    'source_block_embedding',
    'source_unit_embedding',
)
AWAIT_INDEX = 'CALL db.awaitIndex($index_name, $timeout_seconds)'


async def ensure_structural_schema(
    session_factory: Callable[[], AsyncSession],
) -> None:
    """Create KMS2 structural constraints without rewriting stored data."""
    async with session_factory() as session:
        for statement in SCHEMA_STATEMENTS:
            result = await session.run(statement)
            if result is not None:
                await result.consume()


async def ensure_schema(
    session_factory: Callable[[], AsyncSession],
    *,
    embedding_dimension: int,
) -> None:
    """Create structural and vector schema for the selected embedding dimension."""
    await ensure_structural_schema(session_factory)
    async with session_factory() as session:
        result = await session.run(
            """
            SHOW INDEXES YIELD name, type, options
            WHERE name IN $names
            RETURN name, type, options
            """,
            names=list(VECTOR_INDEX_NAMES),
        )
        for index in await result.data():
            dimension = (
                index['options'].get('indexConfig', {}).get('vector.dimensions')
            )
            if index['type'] != 'VECTOR' or dimension != embedding_dimension:
                raise ValueError(
                    f'Embedding index {index["name"]} has dimension {dimension}; '
                    f'selected embeddings require {embedding_dimension}. '
                    'Re-embed stored material and explicitly rebuild the vector '
                    'indexes before changing embedding models.'
                )
        result = await session.run(
            """
            MATCH (node)
            WHERE node.embedding IS NOT NULL
              AND any(label IN labels(node) WHERE label IN $labels)
              AND size(node.embedding) <> $dimension
            RETURN labels(node) AS labels, size(node.embedding) AS dimension
            LIMIT 1
            """,
            labels=['SourceBlock', 'SourceUnit'],
            dimension=embedding_dimension,
        )
        incompatible = await result.data()
        if incompatible:
            raise ValueError(
                f'Stored embeddings have dimension {incompatible[0]["dimension"]}; '
                f'selected embeddings require {embedding_dimension}. '
                'Re-embed stored material before changing embedding models.'
            )
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
