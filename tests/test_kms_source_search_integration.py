"""Opt-in Neo4j tests for user-scoped source retrieval and evidence reads."""

import asyncio
import hashlib
import os
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image

from kms import application
from kms.config.settings import Settings
from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_pedagogical.instruction import Instruction
from kms.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms.core.model.visual_asset import VisualAsset
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.inference.embedding.budget import VOYAGE_EMBEDDING_DIMENSION

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


def _embedding(first: float, second: float, dimension: int) -> list[float]:
    return [first, second, *([0.0] * (dimension - 2))]


def _write_png(path, color: str) -> bytes:
    Image.new('RGB', (3, 2), color).save(path)
    return path.read_bytes()


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_kms_owned_source_search_and_evidence_reading(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = SourceBlockRepository(database.session)
        graph_repository = SourceGraphRepository(database.session)
        prefix = f'kms-source-search-it-{uuid4().hex}'
        user_uuid = f'{prefix}-user'
        foreign_user_uuid = f'{prefix}-foreign-user'
        dimension = VOYAGE_EMBEDDING_DIMENSION

        owned_first = Source(uuid=f'{prefix}-owned-first', key='01-owned.pdf')
        owned_second = Source(uuid=f'{prefix}-owned-second', key='02-owned.pdf')
        foreign_source = Source(uuid=f'{prefix}-foreign', key='foreign.pdf')
        overlap_source = Source(uuid=f'{prefix}-overlap', key='03-overlap.pdf')
        group_source = Source(uuid=f'{prefix}-groups', key='04-groups.pdf')
        sources = [
            owned_first,
            owned_second,
            foreign_source,
            overlap_source,
            group_source,
        ]

        first_block = SourceBlock(
            uuid=f'{prefix}-owned-block-1',
            block_type='paragraph',
            content='Owned candidate one',
            embedding=_embedding(0.8, 0.6, dimension),
        )
        second_block = SourceBlock(
            uuid=f'{prefix}-owned-block-2',
            block_type='paragraph',
            content='Owned candidate two',
            embedding=_embedding(0.6, 0.8, dimension),
        )
        foreign_block = SourceBlock(
            uuid=f'{prefix}-foreign-block',
            block_type='paragraph',
            content='Foreign exact vector match',
            embedding=_embedding(1.0, 0.0, dimension),
        )
        overlap_block = SourceBlock(
            uuid=f'{prefix}-overlap-block',
            block_type='paragraph',
            content='Block shared by two pages',
            embedding=_embedding(0.0, 1.0, dimension),
        )
        overlap_context = SourceBlock(
            uuid=f'{prefix}-overlap-context',
            block_type='paragraph',
            content='Second page context',
        )

        original_one_path = tmp_path / 'original-one.png'
        original_two_path = tmp_path / 'original-two.png'
        crop_path = tmp_path / 'crop.png'
        original_one_bytes = _write_png(original_one_path, 'red')
        original_two_bytes = _write_png(original_two_path, 'green')
        crop_bytes = _write_png(crop_path, 'blue')
        asset_one = VisualAsset(
            uuid=f'{prefix}-asset-one',
            path=str(original_one_path),
            content_sha256=hashlib.sha256(original_one_bytes).hexdigest(),
        )
        asset_two = VisualAsset(
            uuid=f'{prefix}-asset-two',
            path=str(original_two_path),
            content_sha256=hashlib.sha256(original_two_bytes).hexdigest(),
        )

        group_blocks = [
            SourceBlock(
                uuid=f'{prefix}-group-block-{index}',
                block_type='paragraph',
                content=f'Group evidence block {index}',
                crop_path=str(crop_path) if index == 0 else None,
                assets=[asset_one, asset_two] if index == 0 else [],
            )
            for index in range(4)
        ]
        group_block_uuids = [block.uuid for block in group_blocks]
        statement_uuid = f'{prefix}-statement'
        procedure_uuid = f'{prefix}-procedure'
        instruction_uuid = f'{prefix}-instruction'

        source_pages = [
            [SourcePage(index=0, blocks=[first_block])],
            [SourcePage(index=0, blocks=[second_block])],
            [SourcePage(index=0, blocks=[foreign_block])],
            [
                SourcePage(index=7, blocks=[overlap_block]),
                SourcePage(index=8, blocks=[overlap_context]),
            ],
            [
                SourcePage(index=10, blocks=group_blocks[:2]),
                SourcePage(index=11, blocks=group_blocks[2:]),
            ],
        ]

        async def clean_fixture() -> None:
            for source in sources:
                await graph_repository.replace_source(source, [])
                async with database.session() as session:
                    result = await session.run(
                        'MATCH (source:Source {uuid: $uuid}) DETACH DELETE source',
                        uuid=source.uuid,
                    )
                    await result.consume()
            async with database.session() as session:
                result = await session.run(
                    'MATCH (user:User) '
                    'WHERE user.uuid IN $user_uuids '
                    'DETACH DELETE user',
                    user_uuids=[user_uuid, foreign_user_uuid],
                )
                await result.consume()

        async def graph_counts() -> tuple[int, int]:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (node) '
                    'WITH count(node) AS node_count '
                    'OPTIONAL MATCH ()-[relationship]->() '
                    'RETURN node_count, count(relationship) AS relationship_count'
                )
                record = await result.single(strict=True)
            return record['node_count'], record['relationship_count']

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $user_uuid}), '
                    '(:User {uuid: $foreign_user_uuid})',
                    user_uuid=user_uuid,
                    foreign_user_uuid=foreign_user_uuid,
                )
                await result.consume()

            for source, pages in zip(sources, source_pages, strict=True):
                await graph_repository.replace_source(source, pages)

            async with database.session() as session:
                result = await session.run(
                    'UNWIND $ownership AS row '
                    'MATCH (user:User {uuid: row.user_uuid}) '
                    'MATCH (source:Source {uuid: row.source_uuid}) '
                    'CREATE (user)-[:OWNS_SOURCE]->(source)',
                    ownership=[
                        {'user_uuid': user_uuid, 'source_uuid': source.uuid}
                        for source in [
                            owned_first,
                            owned_second,
                            overlap_source,
                            group_source,
                        ]
                    ]
                    + [
                        {
                            'user_uuid': foreign_user_uuid,
                            'source_uuid': foreign_source.uuid,
                        }
                    ],
                )
                await result.consume()
                result = await session.run(
                    'MATCH (source:Source {uuid: $source_uuid})'
                    '-[:HAS_PAGE]->(page:SourcePage {index: 8}) '
                    'MATCH (block:SourceBlock {uuid: $block_uuid}) '
                    'MERGE (page)-[:CONTAINS_BLOCK]->(block)',
                    source_uuid=overlap_source.uuid,
                    block_uuid=overlap_block.uuid,
                )
                await result.consume()

            await graph_repository.replace_pedagogical_material(
                group_source.uuid,
                source_pages[-1],
                [
                    Instruction(
                        uuid=instruction_uuid,
                        member_block_uuids=group_block_uuids[1:3],
                        governed_statement_uuids=[statement_uuid],
                    )
                ],
                [
                    StatementDraft(
                        uuid=statement_uuid,
                        content='A cross-page statement.',
                        member_block_uuids=group_block_uuids[:3],
                    )
                ],
                [
                    ProcedureDraft(
                        uuid=procedure_uuid,
                        content='A procedure sharing a statement block.',
                        member_block_uuids=group_block_uuids[2:],
                    )
                ],
            )

            before_counts = await graph_counts()
            query_embedding = _embedding(1.0, 0.0, dimension)
            candidates = await repository.search_owned_blocks(
                user_uuid,
                query_embedding,
                candidate_limit=4,
            )
            assert [candidate.block.uuid for candidate in candidates[:2]] == [
                first_block.uuid,
                second_block.uuid,
            ]
            assert all(
                candidate.source.uuid != foreign_source.uuid
                for candidate in candidates
            )

            restricted = await repository.search_owned_blocks(
                user_uuid,
                query_embedding,
                candidate_limit=2,
                source_uuids=[owned_second.uuid],
            )
            assert [candidate.block.uuid for candidate in restricted] == [
                second_block.uuid
            ]
            assert (
                await repository.search_owned_blocks(
                    user_uuid,
                    query_embedding,
                    candidate_limit=2,
                    source_uuids=[],
                )
                == []
            )
            assert (
                await repository.search_owned_blocks(
                    user_uuid,
                    query_embedding,
                    candidate_limit=2,
                    source_uuids=[foreign_source.uuid],
                )
                == []
            )

            overlap_candidates = await repository.search_owned_blocks(
                user_uuid,
                _embedding(0.0, 1.0, dimension),
                candidate_limit=4,
                source_uuids=[overlap_source.uuid],
            )
            assert [
                candidate.block.uuid for candidate in overlap_candidates
            ] == [overlap_block.uuid]
            assert overlap_candidates[0].page_indexes == [7, 8]

            ordered_passages = await application.read_source_passages(
                settings,
                user_uuid,
                [second_block.uuid, first_block.uuid],
            )
            assert [passage.source.uuid for passage in ordered_passages] == [
                owned_first.uuid,
                owned_second.uuid,
            ]

            passages = await application.read_source_passages(
                settings,
                user_uuid,
                [group_block_uuids[0]],
            )
            assert len(passages) == 1
            assert [item.block.uuid for item in passages[0].blocks] == (
                group_block_uuids[:3]
            )
            groups = {group.kind: group for group in passages[0].groups}
            assert set(groups) == {'instruction', 'statement'}
            assert (
                groups['statement'].member_block_uuids == group_block_uuids[:3]
            )
            assert (
                groups['instruction'].member_block_uuids
                == group_block_uuids[1:3]
            )
            assert groups['instruction'].governed_statement_uuids == [
                statement_uuid
            ]
            assert passages[0].blocks[0].block.assets == [asset_one, asset_two]
            assert passages[0].blocks[0].block.crop_path == str(crop_path)
            assert group_block_uuids[3] not in {
                item.block.uuid for item in passages[0].blocks
            }

            mixed_passages = await application.read_source_passages(
                settings,
                user_uuid,
                [group_block_uuids[2]],
            )
            assert {group.kind for group in mixed_passages[0].groups} == {
                'instruction',
                'procedure',
                'statement',
            }
            assert [item.block.uuid for item in mixed_passages[0].blocks] == (
                group_block_uuids
            )

            neighbor_passages = await application.read_source_passages(
                settings,
                user_uuid,
                [group_block_uuids[1]],
                expand_memberships=False,
                before=1,
                after=1,
            )
            assert [
                item.block.uuid for item in neighbor_passages[0].blocks
            ] == (group_block_uuids[:3])
            source_end_passages = await application.read_source_passages(
                settings,
                user_uuid,
                [group_block_uuids[3]],
                expand_memberships=False,
                after=1,
            )
            assert [
                item.block.uuid for item in source_end_passages[0].blocks
            ] == [group_block_uuids[3]]

            assert (
                await application.read_source_passages(
                    settings,
                    foreign_user_uuid,
                    [group_block_uuids[0]],
                )
                == []
            )
            images = await application.read_source_images(
                settings,
                user_uuid,
                [asset_two.uuid, asset_one.uuid, asset_two.uuid],
            )
            assert [image.asset.uuid for image in images] == [
                asset_two.uuid,
                asset_one.uuid,
            ]
            assert [image.data for image in images] == [
                original_two_bytes,
                original_one_bytes,
            ]
            assert crop_bytes not in [image.data for image in images]
            with Image.open(BytesIO(images[0].data)) as image:
                assert image.size == (3, 2)
            assert (
                await application.read_source_images(
                    settings,
                    foreign_user_uuid,
                    [asset_one.uuid],
                )
                == []
            )

            assert await graph_counts() == before_counts
        finally:
            await clean_fixture()
            await database.close()

    asyncio.run(exercise())
