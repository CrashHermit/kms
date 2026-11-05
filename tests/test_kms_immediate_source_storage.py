"""Opt-in Neo4j checks for immediately searchable admitted sources."""

import asyncio
import hashlib
import os
from uuid import uuid4

import pytest
from PIL import Image

from kms.config.settings import Settings
from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.visual_asset import VisualAsset
from kms.core.source_snapshot import finalize_snapshot_identities
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms.inference.embedding.budget import VOYAGE_EMBEDDING_DIMENSION
from kms.service.source_evidence import SourceEvidenceService

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
def test_admitted_sources_are_searchable_and_readable_before_teaching(
    tmp_path,
):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        snapshots = SourceSnapshotRepository(database.session)
        blocks = SourceBlockRepository(database.session)
        evidence = SourceEvidenceService(blocks)
        user_uuid = str(uuid4())
        dimension = VOYAGE_EMBEDDING_DIMENSION
        vector = [1.0, *([0.0] * (dimension - 1))]

        image_path = tmp_path / 'diagram.png'
        with Image.new('RGB', (2, 2), 'blue') as image:
            image.save(image_path)
        image_bytes = image_path.read_bytes()
        asset = VisualAsset(
            path=str(image_path),
            content_sha256=hashlib.sha256(image_bytes).hexdigest(),
        )
        first_source = Source(
            uuid=str(uuid4()),
            key='first source',
            metadata={
                'embedding_profile': {
                    'model': 'integration-test',
                    'dimension': dimension,
                }
            },
        )
        first = finalize_snapshot_identities(
            first_source,
            [
                SourcePage(
                    index=0,
                    markdown='First text. Second text. Diagram.',
                    blocks=[
                        SourceBlock(
                            block_type='paragraph',
                            content='First relevant text.',
                            embedding=vector,
                        ),
                        SourceBlock(
                            block_type='paragraph',
                            content='Second relevant text.',
                            embedding=vector,
                        ),
                        SourceBlock(
                            block_type='image',
                            content='Diagram of the topic.',
                            embedding=vector,
                            assets=[asset],
                        ),
                    ],
                )
            ],
        )
        first.source.metadata['normalization_block_positions'] = {
            first.pages[0].blocks[0].uuid: 7
        }
        second_source = Source(
            uuid=str(uuid4()),
            key='second source',
            metadata=first.source.metadata,
        )
        second = finalize_snapshot_identities(
            second_source,
            [
                SourcePage(
                    index=0,
                    markdown='Another source.',
                    blocks=[
                        SourceBlock(
                            block_type='paragraph',
                            content='Another relevant source.',
                            embedding=vector,
                        )
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
                await snapshots.create_owned_snapshot(
                    user_uuid, snapshot.source, snapshot.pages
                )
                for snapshot in (first, second)
            ]
            reused = await snapshots.create_owned_snapshot(
                user_uuid, first.source, first.pages
            )
            assert reused == persisted[0]

            found = await blocks.search_owned_blocks(
                user_uuid,
                vector,
                candidate_limit=10,
            )
            assert {hit.source.uuid for hit in found} == set(source_uuids)
            assert {hit.block.content for hit in found} >= {
                'First relevant text.',
                'Second relevant text.',
                'Diagram of the topic.',
                'Another relevant source.',
            }

            persisted_asset = first.pages[0].blocks[2].assets[0]
            image = await evidence.read_images_at_revisions(
                user_uuid,
                [(first.source.uuid, 0, persisted_asset.uuid)],
            )
            assert [item.asset.uuid for item in image] == [persisted_asset.uuid]
            assert image[0].data == image_bytes
            graph = SourceGraphRepository(database.session)
            parent = first.pages[0].blocks[0]
            child = parent.model_copy(
                update={'uuid': str(uuid4()), 'origin_uuid': parent.uuid}
            )
            split_pages = [
                first.pages[0].model_copy(
                    update={'blocks': [child, *first.pages[0].blocks[1:]]}
                )
            ]
            for _ in range(2):
                await graph.replace_pedagogical_material(
                    first.source.uuid, split_pages, [], [], []
                )
                updated = await snapshots.load_owned_snapshot(
                    user_uuid, first.source.uuid
                )
                assert updated is not None
                assert updated.source.metadata[
                    'normalization_block_positions'
                ] == {child.uuid: 7}
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $sources AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
                    DETACH DELETE asset, block, page, source
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
