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
    CREATE CONSTRAINT visual_asset_uuid IF NOT EXISTS
    FOR (asset:VisualAsset) REQUIRE asset.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_entity_uuid IF NOT EXISTS
    FOR (entity:SourceEntity) REQUIRE entity.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT source_entity_hub_uuid IF NOT EXISTS
    FOR (hub:SourceEntityHub) REQUIRE hub.uuid IS UNIQUE
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
    CREATE CONSTRAINT source_flashcard_uuid IF NOT EXISTS
    FOR (card:SourceFlashcard) REQUIRE card.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_uuid IF NOT EXISTS
    FOR (user:User) REQUIRE user.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT deck_uuid IF NOT EXISTS
    FOR (deck:Deck) REQUIRE deck.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_scheduler_settings_uuid IF NOT EXISTS
    FOR (settings:UserSchedulerSettings) REQUIRE settings.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_card_review_uuid IF NOT EXISTS
    FOR (review:UserCardReview) REQUIRE review.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT review_event_uuid IF NOT EXISTS
    FOR (event:ReviewEvent) REQUIRE event.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT teaching_session_uuid IF NOT EXISTS
    FOR (session:TeachingSession) REQUIRE session.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT teaching_turn_uuid IF NOT EXISTS
    FOR (turn:TeachingTurn) REQUIRE turn.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT probe_assessment_uuid IF NOT EXISTS
    FOR (assessment:ProbeAssessment) REQUIRE assessment.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT teaching_plan_uuid IF NOT EXISTS
    FOR (plan:TeachingPlan) REQUIRE plan.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT teaching_plan_step_uuid IF NOT EXISTS
    FOR (step:TeachingPlanStep) REQUIRE step.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT teaching_question_uuid IF NOT EXISTS
    FOR (question:TeachingQuestion) REQUIRE question.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT proposition_uuid IF NOT EXISTS
    FOR (proposition:Proposition) REQUIRE proposition.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT learner_observation_uuid IF NOT EXISTS
    FOR (observation:LearnerObservation) REQUIRE observation.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT argument_uuid IF NOT EXISTS
    FOR (argument:Argument) REQUIRE argument.uuid IS UNIQUE
    """,
    """
    CREATE CONSTRAINT knowledge_assertion_uuid IF NOT EXISTS
    FOR (assertion:KnowledgeAssertion) REQUIRE assertion.uuid IS UNIQUE
    """,
    """
    CREATE INDEX user_card_review_due_at IF NOT EXISTS
    FOR (review:UserCardReview) ON (review.due_at)
    """,
    # Native GDS projections require these schema keys even with zero edges.
    "CALL db.createRelationshipType('SIMILAR_TO')",
    "CALL db.createProperty('score')",
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
            ('argument_embedding', 'Argument'),
            ('proposition_embedding', 'Proposition'),
            ('source_block_embedding', 'SourceBlock'),
            ('source_entity_embedding', 'SourceEntity'),
            ('source_statement_embedding', 'SourceStatement'),
            ('source_procedure_embedding', 'SourceProcedure'),
            ('source_statement_hub_embedding', 'SourceStatementHub'),
            ('source_procedure_hub_embedding', 'SourceProcedureHub'),
        )
    )


VECTOR_INDEX_NAMES = (
    'argument_embedding',
    'proposition_embedding',
    'source_block_embedding',
    'source_entity_embedding',
    'source_statement_embedding',
    'source_procedure_embedding',
    'source_statement_hub_embedding',
    'source_procedure_hub_embedding',
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
            labels=[
                'Argument',
                'SourceBlock',
                'Proposition',
                'SourceEntity',
                'SourceStatement',
                'SourceProcedure',
                'SourceStatementHub',
                'SourceProcedureHub',
                'LearnerObservation',
            ],
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
