"""Real Neo4j coverage for indexed and scoped block/unit search."""

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kms.config.services import DatabaseSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.media.budget import ImagePayloadBudget
from kms.source.models.snapshot import finalize_snapshot_identities
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.read import SourceReader
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

_REQUIRED = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


class _TextCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_indexed_block_and_unit_search_preserve_ownership_and_filters():
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
        snapshots = []
        for label, embedding in (('owned', near), ('foreign', vector)):
            block = SourceBlock(
                block_type=BlockType.MARKDOWN,
                content='source passage',
                embedding=embedding,
            )
            unit = SourceUnit(
                kind=SourceUnitKind.EXERCISE,
                blocks=[block],
                embedding=embedding,
            )
            source = Source(
                uuid=str(uuid4()),
                key=f'{prefix}-{label}',
                metadata={'embedding_profile': profile},
            )
            snapshots.append(finalize_snapshot_identities(source, [unit]))

        context_snapshot = finalize_snapshot_identities(
            Source(uuid=str(uuid4()), key=f'{prefix}-context'),
            [
                SourceUnit(
                    kind=SourceUnitKind.CONTENT,
                    blocks=[
                        SourceBlock(block_type=BlockType.MARKDOWN, content=text)
                    ],
                )
                for text in ('before', 'target', 'after')
            ],
        )

        source_uuids = [snapshot.source.uuid for snapshot in snapshots]
        source_uuids.append(context_snapshot.source.uuid)
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
            snapshot_repository = SourceSnapshotRepository(database.session)
            await snapshot_repository.create_owned_snapshot(owner, snapshots[0])
            await snapshot_repository.create_owned_snapshot(
                owner, context_snapshot
            )
            await snapshot_repository.create_owned_snapshot(other, snapshots[1])

            repository = SourceRetrievalRepository(database.session)
            reader = SourceReader(repository)
            target_uuid = context_snapshot.units[1].uuid
            window = await reader.read_unit_window(
                owner,
                target_uuid,
                before_tokens=100,
                after_tokens=100,
                budget=TokenBudget(
                    counter=MaterialTokenEstimator(_TextCounter()),
                    token_limit=10000,
                ),
                image_budget=ImagePayloadBudget(
                    max_images=5, max_bytes=10000, max_pixels=10000
                ),
                request_inputs=lambda window: {
                    'before': window.before,
                    'target': window.target,
                    'after': window.after,
                },
            )
            assert window is not None
            assert window.source.uuid == context_snapshot.source.uuid
            assert [unit.uuid for unit in window.before] == [
                context_snapshot.units[0].uuid
            ]
            assert window.target.uuid == target_uuid
            assert [unit.uuid for unit in window.after] == [
                context_snapshot.units[2].uuid
            ]
            assert (
                await reader.read_unit_window(
                    other,
                    target_uuid,
                    before_tokens=100,
                    after_tokens=100,
                    budget=TokenBudget(
                        counter=MaterialTokenEstimator(_TextCounter()),
                        token_limit=10000,
                    ),
                    image_budget=ImagePayloadBudget(
                        max_images=5, max_bytes=10000, max_pixels=10000
                    ),
                    request_inputs=lambda window: {'target': window.target},
                )
                is None
            )
            blocks = await repository.search_owned_blocks(
                owner, vector, candidate_limit=10
            )
            units = await repository.search_owned_units(
                owner, vector, candidate_limit=10
            )
            assert [item.block.uuid for item in blocks] == [
                snapshots[0].units[0].blocks[0].uuid
            ]
            assert blocks[0].unit_uuid == snapshots[0].units[0].uuid
            assert [item.unit.uuid for item in units] == [
                snapshots[0].units[0].uuid
            ]
            assert units[0].unit.kind is SourceUnitKind.EXERCISE
            assert [block.uuid for block in units[0].unit.blocks] == [
                snapshots[0].units[0].blocks[0].uuid
            ]
            assert units[0].unit.invalidated_at is None
            active = await repository.read_units(
                owner, [snapshots[0].units[0].uuid]
            )
            assert active[0].unit.invalidated_at is None
            foreign_blocks = await repository.search_owned_blocks(
                other, vector, candidate_limit=10
            )
            foreign_units = await repository.search_owned_units(
                other, vector, candidate_limit=10
            )
            assert [hit.block.uuid for hit in foreign_blocks] == [
                snapshots[1].units[0].blocks[0].uuid
            ]
            assert [hit.unit.uuid for hit in foreign_units] == [
                snapshots[1].units[0].uuid
            ]

            scoped = [snapshots[0].source.uuid]
            filtered_blocks = await repository.search_owned_blocks(
                owner, vector, candidate_limit=1, source_uuids=scoped
            )
            filtered_units = await repository.search_owned_units(
                owner,
                vector,
                candidate_limit=1,
                source_uuids=scoped,
                kind=SourceUnitKind.EXERCISE,
            )
            assert [item.block.uuid for item in filtered_blocks] == [
                snapshots[0].units[0].blocks[0].uuid
            ]
            assert [item.unit.uuid for item in filtered_units] == [
                snapshots[0].units[0].uuid
            ]
            assert filtered_units[0].unit.invalidated_at is None

            invalidated_at = datetime(2026, 1, 2, tzinfo=UTC)
            async with database.session() as session:
                result = await session.run(
                    'MATCH (unit:SourceUnit {uuid: $uuid}) '
                    'SET unit.invalidated_at = $invalidated_at',
                    uuid=snapshots[0].units[0].uuid,
                    invalidated_at=invalidated_at,
                )
                await result.consume()
            invalidated = await repository.read_units(
                owner, [snapshots[0].units[0].uuid]
            )
            assert invalidated[0].unit.invalidated_at == invalidated_at
            filtered_units = await repository.search_owned_units(
                owner, vector, candidate_limit=1, source_uuids=scoped
            )
            assert filtered_units[0].unit.invalidated_at == invalidated_at
            units = await repository.search_owned_units(
                owner, vector, candidate_limit=10
            )
            assert units[0].unit.invalidated_at == invalidated_at
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $source_uuids AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit)
                    OPTIONAL MATCH (unit)-[:HAS_BLOCK]->(block:SourceBlock)
                    DETACH DELETE block, unit, source
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
