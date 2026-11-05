"""Real Neo4j coverage for indexed and scoped semantic retrieval."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms.config.services import DatabaseSettings
from kms.config.settings import Settings
from kms.core.model.knowledge.argument import argument_text
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.database.knowledge.repository import KnowledgeRepository
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.inference.embedding.budget import VOYAGE_EMBEDDING_DIMENSION

_REQUIRED = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_indexed_search_and_exact_filtered_search_preserve_ownership():
    async def exercise():
        settings = Settings(
            _env_file=None,
            database=DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            ),
        )
        database = DatabaseClient(settings.database)
        dimension = VOYAGE_EMBEDDING_DIMENSION
        prefix = f'vector-search-{uuid4().hex}'
        owner = f'{prefix}-owner'
        other = f'{prefix}-other'
        vector = [1.0, *([0.0] * (dimension - 1))]
        near = [0.8, 0.6, *([0.0] * (dimension - 2))]
        entries = [
            {'id': f'{prefix}-owned', 'user': owner, 'embedding': near},
            {'id': f'{prefix}-foreign', 'user': other, 'embedding': vector},
        ]
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $entries AS entry
                    MERGE (user:User {uuid: entry.user})
                    CREATE (user)-[:OWNS_SOURCE]->
                        (source:Source {uuid: entry.id, key: entry.id, revision: 0})
                    CREATE (source)-[:HAS_PAGE]->(page:SourcePage {
                        uuid: entry.id + '-page', index: 0
                    })
                    CREATE (page)-[:CONTAINS_BLOCK]->(block:SourceBlock {
                        uuid: entry.id + '-block', block_type: 'paragraph',
                        content: 'source passage', embedding: entry.embedding
                    })
                    CREATE (source)-[:FIRST_BLOCK]->(block)
                    CREATE (block)-[:MEMBER_OF]->(statement:SourceStatement {
                        uuid: entry.id + '-statement', content: 'statement',
                        is_exercise: true, embedding: entry.embedding
                    })
                    CREATE (block)-[:MEMBER_OF]->(procedure:SourceProcedure {
                        uuid: entry.id + '-procedure', content: 'procedure',
                        embedding: entry.embedding
                    })
                    CREATE (user)-[:HAS_PROPOSITION]->(proposition:Proposition {
                        uuid: entry.id + '-proposition', content: 'proposition',
                        embedding: entry.embedding, created_at: datetime()
                    })
                    CREATE (user)-[:HAS_ARGUMENT]->(argument:Argument {
                        uuid: entry.id + '-argument', explanation: 'Complete argument.',
                        embedding: entry.embedding, created_at: datetime()
                    })
                    CREATE (proposition)-[:PREMISE_OF]->(argument)
                    CREATE (argument)-[:PROVES]->(proposition)
                    CREATE (argument)-[:SUPPORTED_BY]->(block)
                    CREATE (user)-[:HAS_ARGUMENT]->(
                        zero_argument:Argument {
                            uuid: entry.id + '-zero-argument',
                            explanation: 'No premise is required.',
                            embedding: entry.embedding,
                            created_at: datetime()
                        }
                    )
                    CREATE (zero_argument)-[:PROVES]->(proposition)
                    """,
                    entries=entries,
                )
                await result.consume()

            sources = SourceBlockRepository(database.session)
            knowledge = KnowledgeRepository(database.session)
            blocks = await sources.search_owned_blocks(
                owner, vector, candidate_limit=10
            )
            statements = await sources.search_owned_statements(
                owner, vector, candidate_limit=10
            )
            procedures = await sources.search_owned_procedures(
                owner, vector, candidate_limit=10
            )
            propositions = await knowledge.search_propositions(
                owner, vector, candidate_limit=10
            )
            arguments = await knowledge.search_arguments(
                owner, vector, candidate_limit=10
            )
            assert [item.block.uuid for item in blocks] == [
                f'{prefix}-owned-block'
            ]
            assert blocks[0].page_indexes == [0]
            assert [item.statement.uuid for item in statements] == [
                f'{prefix}-owned-statement'
            ]
            assert [item.procedure.uuid for item in procedures] == [
                f'{prefix}-owned-procedure'
            ]
            assert [item.proposition.uuid for item in propositions] == [
                f'{prefix}-owned-proposition'
            ]
            assert [item.record.argument.uuid for item in arguments] == [
                f'{prefix}-owned-argument',
                f'{prefix}-owned-zero-argument',
            ]
            assert arguments[0].record.argument.embedding == near
            assert arguments[0].record.premise_uuids == [
                f'{prefix}-owned-proposition'
            ]
            assert (
                arguments[0].record.conclusion_uuid
                == f'{prefix}-owned-proposition'
            )
            assert arguments[0].record.supporting_block_uuids == [
                f'{prefix}-owned-block'
            ]
            assert arguments[0].content == argument_text(
                ['proposition'],
                'proposition',
                'Complete argument.',
            )
            assert arguments[1].record.premise_uuids == []
            assert (
                await knowledge.search_propositions(
                    f'{prefix}-missing', vector, candidate_limit=10
                )
                == []
            )
            assert (
                await knowledge.search_arguments(
                    f'{prefix}-missing',
                    vector,
                    candidate_limit=10,
                )
                == []
            )
            assert await knowledge.read_arguments(
                owner,
                [f'{prefix}-owned-argument'],
            ) == [arguments[0].record]

            scoped = [f'{prefix}-owned']
            filtered_blocks = await sources.search_owned_blocks(
                owner, vector, candidate_limit=1, source_uuids=scoped
            )
            filtered_statements = await sources.search_owned_statements(
                owner, vector, candidate_limit=1, is_exercise=True
            )
            filtered_procedures = await sources.search_owned_procedures(
                owner, vector, candidate_limit=1, source_uuids=scoped
            )
            assert [item.block.uuid for item in filtered_blocks] == [
                f'{prefix}-owned-block'
            ]
            assert [item.statement.uuid for item in filtered_statements] == [
                f'{prefix}-owned-statement'
            ]
            assert [item.procedure.uuid for item in filtered_procedures] == [
                f'{prefix}-owned-procedure'
            ]
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (node) WHERE node.uuid STARTS WITH $prefix '
                    'DETACH DELETE node',
                    prefix=prefix,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
