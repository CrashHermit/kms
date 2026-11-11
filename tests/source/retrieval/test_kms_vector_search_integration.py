"""Real Neo4j coverage for indexed and scoped semantic retrieval."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms.config.services import DatabaseSettings
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.source.models.material import ProcedureDraft, StatementDraft
from kms.source.models.snapshot import finalize_snapshot_identities
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

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
    async def exercise() -> None:
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
        profile = {'model': 'integration-test', 'dimension': dimension}
        sources = [
            Source(
                uuid=str(uuid4()),
                key=f'{prefix}-{label}',
                metadata={'embedding_profile': profile},
            )
            for label in ('owned', 'foreign')
        ]
        snapshots = []
        for source, embedding in zip(sources, (near, vector), strict=True):
            block = SourceBlock(
                block_type=BlockType.MARKDOWN,
                content='source passage',
                embedding=embedding,
            )
            snapshot = finalize_snapshot_identities(
                source,
                [SourcePage(index=0, blocks=[block])],
                [],
                [
                    StatementDraft(
                        uuid=f'{prefix}-{source.key}-statement',
                        content='statement',
                        is_exercise=True,
                        embedding=embedding,
                        member_block_uuids=[block.uuid],
                    )
                ],
                [
                    ProcedureDraft(
                        content='procedure',
                        embedding=embedding,
                        member_block_uuids=[block.uuid],
                    )
                ],
            )
            snapshots.append(snapshot)

        source_uuids = [snapshot.source.uuid for snapshot in snapshots]
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $owner}), (:User {uuid: $other})',
                    owner=owner,
                    other=other,
                )
                await result.consume()
            repository = SourceSnapshotRepository(database.session)
            await repository.create_owned_snapshot(owner, snapshots[0])
            await repository.create_owned_snapshot(other, snapshots[1])

            sources_repository = SourceRetrievalRepository(database.session)
            blocks = await sources_repository.search_owned_blocks(
                owner, vector, candidate_limit=10
            )
            statements = await sources_repository.search_owned_statements(
                owner, vector, candidate_limit=10
            )
            procedures = await sources_repository.search_owned_procedures(
                owner, vector, candidate_limit=10
            )
            assert [item.block.uuid for item in blocks] == [
                snapshots[0].pages[0].blocks[0].uuid
            ]
            assert blocks[0].page_indexes == [0]
            assert [item.statement.uuid for item in statements] == [
                snapshots[0].statements[0].uuid
            ]
            assert [item.procedure.uuid for item in procedures] == [
                snapshots[0].procedures[0].uuid
            ]

            scoped = [snapshots[0].source.uuid]
            filtered_blocks = await sources_repository.search_owned_blocks(
                owner, vector, candidate_limit=1, source_uuids=scoped
            )
            filtered_statements = (
                await sources_repository.search_owned_statements(
                    owner, vector, candidate_limit=1, is_exercise=True
                )
            )
            filtered_procedures = (
                await sources_repository.search_owned_procedures(
                    owner, vector, candidate_limit=1, source_uuids=scoped
                )
            )
            assert [item.block.uuid for item in filtered_blocks] == [
                snapshots[0].pages[0].blocks[0].uuid
            ]
            assert [item.statement.uuid for item in filtered_statements] == [
                snapshots[0].statements[0].uuid
            ]
            assert [item.procedure.uuid for item in filtered_procedures] == [
                snapshots[0].procedures[0].uuid
            ]
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $source_uuids AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(material)
                    DETACH DELETE material, block, page, source
                    """,
                    source_uuids=source_uuids,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User) WHERE user.uuid IN $users '
                    'DETACH DELETE user',
                    users=[owner, other],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
