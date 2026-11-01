"""Real Neo4j coverage for indexed and scoped semantic retrieval."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms2.composition.embedding import embedding_dimension
from kms2.config.services import DatabaseSettings
from kms2.config.settings import Settings
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.knowledge.repository import KnowledgeRepository
from kms2.database.source.source_block_repository import SourceBlockRepository

_REQUIRED = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS2 Neo4j integration environment is not enabled',
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
        dimension = embedding_dimension(settings)
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
            assert (
                await knowledge.search_propositions(
                    f'{prefix}-missing', vector, candidate_limit=10
                )
                == []
            )

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
