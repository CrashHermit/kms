"""Opt-in Neo4j checks for immediately searchable v5 source snapshots."""

import asyncio
import hashlib
import os
from uuid import uuid4

import pytest
from PIL import Image

from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.source.models.snapshot import finalize_snapshot_identities
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.read import SourceReader
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_owned_v5_units_are_searchable_and_images_remain_block_owned(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        snapshots = SourceSnapshotRepository(database.session)
        repository = SourceRetrievalRepository(database.session)
        reader = SourceReader(repository)
        user_uuid = str(uuid4())
        dimension = VOYAGE_EMBEDDING_DIMENSION
        vector = [1.0, *([0.0] * (dimension - 1))]
        profile = {'model': 'integration-test', 'dimension': dimension}

        image_path = tmp_path / 'diagram.png'
        with Image.new('RGB', (2, 2), 'blue') as image:
            image.save(image_path)
        image_bytes = image_path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        first = finalize_snapshot_identities(
            Source(
                uuid=str(uuid4()),
                key='first source',
                metadata={
                    'embedding_profile': profile,
                },
            ),
            [
                SourceUnit(
                    kind=SourceUnitKind.EXERCISE,
                    embedding=vector,
                    blocks=[
                        SourceBlock(
                            block_type=BlockType.PARAGRAPH,
                            content='First relevant text.',
                            embedding=vector,
                        ),
                        SourceBlock(
                            block_type=BlockType.IMAGE,
                            content='Diagram of the topic.',
                            embedding=vector,
                            image_path=str(image_path),
                            image_sha256=image_hash,
                        ),
                    ],
                )
            ],
        )
        second = finalize_snapshot_identities(
            Source(
                uuid=str(uuid4()),
                key='second source',
                metadata={
                    'embedding_profile': profile,
                },
            ),
            [
                SourceUnit(
                    kind=SourceUnitKind.CONTENT,
                    embedding=vector,
                    blocks=[
                        SourceBlock(
                            block_type=BlockType.PARAGRAPH,
                            content='Another relevant source.',
                            embedding=vector,
                        ),
                    ],
                )
            ],
        )
        source_uuids = [first.source.uuid, second.source.uuid]
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user_uuid
                )
                await result.consume()

            persisted = [
                await snapshots.create_owned_snapshot(user_uuid, snapshot)
                for snapshot in (first, second)
            ]
            assert (
                await snapshots.create_owned_snapshot(user_uuid, first)
                == persisted[0]
            )
            loaded = await snapshots.load_owned_snapshot(
                user_uuid, first.source.uuid
            )
            assert loaded == persisted[0]
            assert loaded.units[0].kind is SourceUnitKind.EXERCISE

            found = await repository.search_owned_blocks(
                user_uuid,
                vector,
                candidate_limit=10,
            )
            assert {hit.source.uuid for hit in found} == set(source_uuids)
            assert {hit.block.content for hit in found} >= {
                'First relevant text.',
                'Diagram of the topic.',
                'Another relevant source.',
            }
            unit_hits = await repository.search_owned_units(
                user_uuid,
                vector,
                candidate_limit=10,
                kind=SourceUnitKind.EXERCISE,
            )
            assert [(hit.source.uuid, hit.unit.kind) for hit in unit_hits] == [
                (first.source.uuid, SourceUnitKind.EXERCISE)
            ]

            image_block = loaded.units[0].blocks[1]
            images = await reader.read_images(user_uuid, [image_block.uuid])
            assert [image.block_uuid for image in images] == [image_block.uuid]
            assert images[0].data == image_bytes
            assert (
                await reader.read_images('foreign-user', [image_block.uuid])
                == []
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $sources AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit)
                    OPTIONAL MATCH (unit)-[:HAS_BLOCK]->(block:SourceBlock)
                    DETACH DELETE block, unit, source
                    """,
                    sources=source_uuids,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $uuid}) DETACH DELETE user',
                    uuid=user_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
