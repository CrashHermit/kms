"""KMS2-only Neo4j schema statements and initialization."""

from collections.abc import Callable

from neo4j import AsyncSession

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
    CREATE CONSTRAINT source_statement_uuid IF NOT EXISTS
    FOR (statement:SourceStatement) REQUIRE statement.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_procedure_uuid IF NOT EXISTS
    FOR (procedure:SourceProcedure) REQUIRE procedure.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_uuid IF NOT EXISTS
    FOR (user:User) REQUIRE user.uuid IS UNIQUE
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
            ('source_statement_embedding', 'SourceStatement'),
            ('source_procedure_embedding', 'SourceProcedure'),
        )
    )


VECTOR_INDEX_NAMES = (
    'source_block_embedding',
    'source_statement_embedding',
    'source_procedure_embedding',
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
            labels=['SourceBlock', 'SourceStatement', 'SourceProcedure'],
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
