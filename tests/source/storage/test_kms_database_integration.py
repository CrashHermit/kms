"""Opt-in Neo4j integration coverage for current KMS2 persistence contracts."""

import asyncio
import hashlib
import os
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image

from kms.config.services import DatabaseSettings
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.media.budget import ImagePayloadBudget
from kms.media.content import collect_image_assets
from kms.media.images import load_content_images
from kms.source.context import SourceBlockWindow, project_source_window
from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.snapshot import (
    SourceSnapshotConflict,
    finalize_snapshot_identities,
)
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.storage.snapshots import SourceSnapshotRepository

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


def _embedding(first: float, second: float, dimension: int) -> list[float]:
    return [first, second, *([0.0] * (dimension - 2))]


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_kms_neo4j_pedagogical_source_round_trip():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        user_uuid = str(uuid4())
        source = Source(
            uuid=str(uuid4()),
            key='pedagogical-source.pdf',
            metadata={
                'embedding_profile': {
                    'model': 'integration-test',
                    'dimension': VOYAGE_EMBEDDING_DIMENSION,
                }
            },
        )
        vector = _embedding(1.0, 0.0, VOYAGE_EMBEDDING_DIMENSION)
        blocks = [
            SourceBlock(
                block_type=BlockType.MARKDOWN,
                content=content,
                embedding=vector,
            )
            for content in (
                'A real square matrix is invertible exactly when its determinant is nonzero.',
                'The matrix A has real entries.',
                'The argument continues after the definition.',
            )
        ]
        source_pages = [SourcePage(index=0, blocks=blocks)]
        snapshot = finalize_snapshot_identities(
            source,
            source_pages,
            [
                Instruction(
                    member_block_uuids=[block.uuid for block in blocks],
                    governed_statement_uuids=['statement'],
                )
            ],
            [
                StatementDraft(
                    uuid='statement',
                    content='An exercise about the real matrix A.',
                    embedding=_embedding(0.1, 0.0, VOYAGE_EMBEDDING_DIMENSION),
                    member_block_uuids=[blocks[1].uuid, blocks[2].uuid],
                    is_exercise=True,
                )
            ],
            [
                ProcedureDraft(
                    content='Read the matrix and determine its dimensions.',
                    embedding=_embedding(0.2, 0.0, VOYAGE_EMBEDDING_DIMENSION),
                    member_block_uuids=[blocks[0].uuid],
                )
            ],
        )
        snapshots = SourceSnapshotRepository(database.session)
        rollback_source_uuid: str | None = None

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=VOYAGE_EMBEDDING_DIMENSION,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user_uuid
                )
                await result.consume()

            persisted = await snapshots.create_owned_snapshot(
                user_uuid, snapshot
            )
            loaded = await snapshots.load_owned_snapshot(user_uuid, source.uuid)
            assert loaded == persisted
            assert loaded.source.created_at == snapshot.source.created_at
            assert [
                block.created_at
                for page in loaded.pages
                for block in page.blocks
            ] == [
                block.created_at
                for page in snapshot.pages
                for block in page.blocks
            ]
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                          -[ownership:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
                    MATCH (source)-[has_page:HAS_PAGE]->(page:SourcePage)
                    MATCH (page)-[contains_block:CONTAINS_BLOCK]->(
                        block:SourceBlock {uuid: $block_uuid}
                    )
                    RETURN source.created_at AS source_created_at,
                           page.created_at AS page_created_at,
                           block.created_at AS block_created_at,
                           ownership.created_at AS ownership_created_at,
                           has_page.created_at AS has_page_created_at,
                           contains_block.created_at AS contains_block_created_at
                    """,
                    user_uuid=user_uuid,
                    source_uuid=source.uuid,
                    block_uuid=snapshot.pages[0].blocks[0].uuid,
                )
                timestamps = await result.single(strict=True)
            assert (
                timestamps['source_created_at'].to_native()
                == snapshot.source.created_at
            )
            assert timestamps['page_created_at'] is not None
            assert (
                timestamps['block_created_at'].to_native()
                == snapshot.pages[0].blocks[0].created_at
            )
            assert timestamps['ownership_created_at'] is not None
            assert timestamps['has_page_created_at'] is not None
            assert timestamps['contains_block_created_at'] is not None
            assert loaded.instructions[0].member_block_uuids == [
                block.uuid for block in loaded.pages[0].blocks
            ]
            assert loaded.instructions[0].governed_statement_uuids == [
                loaded.statements[0].uuid
            ]
            assert loaded.statements[0].member_block_uuids == [
                loaded.pages[0].blocks[1].uuid,
                loaded.pages[0].blocks[2].uuid,
            ]
            assert loaded.procedures[0].member_block_uuids == [
                loaded.pages[0].blocks[0].uuid
            ]
            rollback_source_uuid = str(uuid4())
            rollback_source = Source(
                uuid=rollback_source_uuid,
                key='rollback-source.pdf',
                metadata={
                    'embedding_profile': {
                        'model': 'integration-test',
                        'dimension': VOYAGE_EMBEDDING_DIMENSION,
                    }
                },
            )
            rollback_snapshot = finalize_snapshot_identities(
                rollback_source,
                [
                    SourcePage(
                        index=0,
                        blocks=[
                            SourceBlock(
                                block_type=BlockType.MARKDOWN,
                                content='This write must roll back.',
                                embedding=vector,
                            )
                        ],
                    )
                ],
                [],
                [],
                [],
            )
            rollback_snapshot = rollback_snapshot.model_copy(
                update={
                    'source': rollback_snapshot.source.model_copy(
                        update={
                            'metadata': {
                                **rollback_snapshot.source.metadata,
                                'processed_digest': '0' * 64,
                            }
                        }
                    )
                }
            )
            with pytest.raises(SourceSnapshotConflict):
                await snapshots.create_owned_snapshot(
                    user_uuid, rollback_snapshot
                )
            assert (
                await snapshots.load_owned_snapshot(
                    user_uuid, rollback_source.uuid
                )
                is None
            )

        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $uuids AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(material)
                    DETACH DELETE material, block, page, source
                    """,
                    uuids=[
                        source.uuid,
                        *(
                            [rollback_source_uuid]
                            if rollback_source_uuid is not None
                            else []
                        ),
                    ],
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $uuid}) DETACH DELETE user',
                    uuid=user_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_kms_statement_procedure_material_round_trip():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        user_uuid = str(uuid4())
        source = Source(
            uuid=str(uuid4()),
            key='source-material.pdf',
            metadata={
                'embedding_profile': {
                    'model': 'integration-test',
                    'dimension': VOYAGE_EMBEDDING_DIMENSION,
                }
            },
        )
        blocks = [
            SourceBlock(
                block_type=BlockType.MARKDOWN,
                content=f'Source block {index}.',
                embedding=_embedding(
                    1.0, index / 100, VOYAGE_EMBEDDING_DIMENSION
                ),
            )
            for index in range(2)
        ]
        statements = [
            StatementDraft(
                content=f'A real square matrix statement {index}.',
                embedding=_embedding(
                    1.0, index / 100, VOYAGE_EMBEDDING_DIMENSION
                ),
                member_block_uuids=[block.uuid for block in blocks[:index]],
            )
            for index in (1, 2)
        ]
        procedures = [
            ProcedureDraft(
                content=f'Procedure step sequence {index}.',
                embedding=_embedding(
                    1.0, index / 100, VOYAGE_EMBEDDING_DIMENSION
                ),
                member_block_uuids=[block.uuid for block in blocks[:index]],
            )
            for index in (1, 2)
        ]
        snapshot = finalize_snapshot_identities(
            source,
            [SourcePage(index=0, blocks=blocks)],
            [],
            statements,
            procedures,
        )
        snapshots = SourceSnapshotRepository(database.session)
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=VOYAGE_EMBEDDING_DIMENSION,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user_uuid
                )
                await result.consume()
            await snapshots.create_owned_snapshot(user_uuid, snapshot)
            loaded = await snapshots.load_owned_snapshot(user_uuid, source.uuid)
            assert loaded is not None
            assert [item.uuid for item in loaded.statements] == [
                item.uuid for item in snapshot.statements
            ]
            assert [item.member_block_uuids for item in loaded.statements] == [
                [loaded.pages[0].blocks[0].uuid],
                [
                    loaded.pages[0].blocks[0].uuid,
                    loaded.pages[0].blocks[1].uuid,
                ],
            ]
            assert [item.uuid for item in loaded.procedures] == [
                item.uuid for item in snapshot.procedures
            ]
            assert [item.member_block_uuids for item in loaded.procedures] == [
                [loaded.pages[0].blocks[0].uuid],
                [
                    loaded.pages[0].blocks[0].uuid,
                    loaded.pages[0].blocks[1].uuid,
                ],
            ]
            assert 'source_uuid' not in loaded.statements[0].model_dump()
            assert 'source_uuid' not in loaded.procedures[0].model_dump()
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(material)
                    DETACH DELETE material, block, page, source
                    """,
                    uuid=source.uuid,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $uuid}) DETACH DELETE user',
                    uuid=user_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_kms_source_direct_images_round_trip(tmp_path):
    async def exercise() -> None:
        settings = Settings(
            _env_file=None,
            database=DatabaseSettings(
                uri=os.environ['KMS_DATABASE__URI'],
                username=os.environ['KMS_DATABASE__USERNAME'],
                password=os.environ['KMS_DATABASE__PASSWORD'],
                database=os.environ['KMS_DATABASE__DATABASE'],
            ),
        )
        database = DatabaseClient(settings.database)
        user_uuid = str(uuid4())
        source = Source(
            uuid=str(uuid4()),
            key='direct-images.pdf',
            metadata={
                'embedding_profile': {
                    'model': 'integration-test',
                    'dimension': VOYAGE_EMBEDDING_DIMENSION,
                }
            },
        )
        red_path = tmp_path / 'red.png'
        blue_path = tmp_path / 'blue.png'
        Image.new('RGB', (8, 8), (255, 0, 0)).save(red_path, format='PNG')
        Image.new('RGB', (8, 8), (0, 0, 255)).save(blue_path, format='PNG')
        image_blocks = [
            SourceBlock(
                block_type=BlockType.IMAGE,
                content='Red figure.',
                image_path=str(red_path),
                image_sha256=hashlib.sha256(red_path.read_bytes()).hexdigest(),
                image_page_indexes=[0],
            ),
            SourceBlock(
                block_type=BlockType.IMAGE,
                content='Blue figure.',
                image_path=str(blue_path),
                image_sha256=hashlib.sha256(blue_path.read_bytes()).hexdigest(),
                image_page_indexes=[0],
            ),
        ]
        text_block = SourceBlock(
            block_type=BlockType.MARKDOWN,
            content='The figure has a red circle and a blue square.',
        )
        snapshot = finalize_snapshot_identities(
            source,
            [SourcePage(index=0, blocks=[*image_blocks, text_block])],
            [],
            [],
            [],
        )
        snapshots = SourceSnapshotRepository(database.session)
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=VOYAGE_EMBEDDING_DIMENSION,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user_uuid
                )
                await result.consume()
            await snapshots.create_owned_snapshot(user_uuid, snapshot)
            loaded = await snapshots.load_owned_snapshot(user_uuid, source.uuid)
            assert loaded is not None
            assert [
                (block.image_path, block.image_sha256, block.image_page_indexes)
                for block in loaded.pages[0].blocks
            ] == [
                (
                    block.image_path,
                    block.image_sha256,
                    block.image_page_indexes,
                )
                for block in snapshot.pages[0].blocks
            ]

            projected = project_source_window(
                SourceBlockWindow(target=loaded.pages[0].blocks)
            )
            assert [
                len(block.material.images) for block in projected.target
            ] == [1, 1, 0]
            native_images = await load_content_images(
                collect_image_assets(projected),
                budget=ImagePayloadBudget(
                    max_images=2,
                    max_bytes=1_000_000,
                    max_pixels=128,
                ),
            )
            with Image.open(BytesIO(native_images[0])) as image:
                assert image.getpixel((0, 0)) == (255, 0, 0)
            with Image.open(BytesIO(native_images[1])) as image:
                assert image.getpixel((0, 0)) == (0, 0, 255)
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(material)
                    DETACH DELETE material, block, page, source
                    """,
                    uuid=source.uuid,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $uuid}) DETACH DELETE user',
                    uuid=user_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
