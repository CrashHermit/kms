"""Opt-in Neo4j tests for user-scoped source retrieval and evidence reads."""

import asyncio
import hashlib
import os
from io import BytesIO
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError
from PIL import Image

from kms import application
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.snapshot import finalize_snapshot_identities
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

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
        repository = SourceRetrievalRepository(database.session)
        snapshots = SourceSnapshotRepository(database.session)
        prefix = f'kms-source-search-it-{uuid4().hex}'
        user_uuid = f'{prefix}-user'
        foreign_user_uuid = f'{prefix}-foreign-user'
        dimension = VOYAGE_EMBEDDING_DIMENSION
        profile = {'model': 'integration-test', 'dimension': dimension}

        sources = [
            Source(
                uuid=str(uuid4()),
                key=key,
                metadata={'embedding_profile': profile},
            )
            for key in (
                '01-owned.pdf',
                '02-owned.pdf',
                'foreign.pdf',
                '03-overlap.pdf',
                '04-groups.pdf',
            )
        ]
        (
            owned_first,
            owned_second,
            foreign_source,
            overlap_source,
            group_source,
        ) = sources

        first_block = SourceBlock(
            block_type=BlockType.MARKDOWN,
            content='Owned candidate one',
            embedding=_embedding(0.8, 0.6, dimension),
        )
        second_block = SourceBlock(
            block_type=BlockType.MARKDOWN,
            content='Owned candidate two',
            embedding=_embedding(0.6, 0.8, dimension),
        )
        foreign_block = SourceBlock(
            block_type=BlockType.MARKDOWN,
            content='Foreign exact vector match',
            embedding=_embedding(1.0, 0.0, dimension),
        )
        overlap_path = tmp_path / 'overlap.png'
        overlap_bytes = _write_png(overlap_path, 'blue')
        overlap_block = SourceBlock(
            block_type=BlockType.IMAGE,
            content='Block shared by two pages',
            embedding=_embedding(0.0, 1.0, dimension),
            image_path=str(overlap_path),
            image_sha256=hashlib.sha256(overlap_bytes).hexdigest(),
            image_page_indexes=[7, 8],
        )
        overlap_context = SourceBlock(
            block_type=BlockType.MARKDOWN,
            content='Second page context',
        )

        image_paths = [
            tmp_path / 'original-one.png',
            tmp_path / 'original-two.png',
        ]
        image_bytes = [
            _write_png(image_paths[0], 'red'),
            _write_png(image_paths[1], 'green'),
        ]
        group_blocks = [
            SourceBlock(
                block_type=(
                    BlockType.IMAGE if index in (0, 2) else BlockType.MARKDOWN
                ),
                content=f'Group evidence block {index}',
                image_path=str(image_paths[index // 2])
                if index in (0, 2)
                else None,
                image_sha256=(
                    hashlib.sha256(image_bytes[index // 2]).hexdigest()
                    if index in (0, 2)
                    else None
                ),
                image_page_indexes=[10 + index // 2] if index in (0, 2) else [],
            )
            for index in range(4)
        ]
        group_block_uuids = [block.uuid for block in group_blocks]
        instruction = Instruction(
            member_block_uuids=group_block_uuids[1:3],
            governed_statement_uuids=['statement'],
        )
        statement = StatementDraft(
            uuid='statement',
            content='A cross-page statement.',
            embedding=_embedding(0.9, 0.1, dimension),
            member_block_uuids=group_block_uuids[:3],
        )
        procedure = ProcedureDraft(
            content='A procedure sharing a statement block.',
            embedding=_embedding(0.7, 0.3, dimension),
            member_block_uuids=group_block_uuids[2:],
        )
        page_sets = [
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
        snapshots_to_create = [
            finalize_snapshot_identities(source, pages, [], [], [])
            for source, pages in zip(sources[:4], page_sets[:4], strict=True)
        ]
        group_snapshot = finalize_snapshot_identities(
            group_source,
            page_sets[-1],
            [instruction],
            [statement],
            [procedure],
        )
        snapshots_to_create.append(group_snapshot)
        first, second, foreign, overlap, group = snapshots_to_create
        first_block, second_block = (
            first.pages[0].blocks[0],
            second.pages[0].blocks[0],
        )
        foreign_block = foreign.pages[0].blocks[0]
        overlap_block = overlap.pages[0].blocks[0]
        group_block_uuids = [
            block.uuid for page in group.pages for block in page.blocks
        ]
        statement_uuid = group.statements[0].uuid
        procedure_uuid = group.procedures[0].uuid

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

            for snapshot, owner in zip(
                snapshots_to_create,
                [user_uuid, user_uuid, foreign_user_uuid, user_uuid, user_uuid],
                strict=True,
            ):
                await snapshots.create_owned_snapshot(owner, snapshot)

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
            assert [item.block.uuid for item in overlap_candidates] == [
                overlap_block.uuid
            ]
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
            assert passages[0].blocks[0].block.image_path == str(image_paths[0])
            assert (
                passages[0].blocks[0].block.image_sha256
                == hashlib.sha256(image_bytes[0]).hexdigest()
            )
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
            assert procedure_uuid in {
                group.uuid for group in mixed_passages[0].groups
            }

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
                    settings, foreign_user_uuid, [group_block_uuids[0]]
                )
                == []
            )
            images = await application.read_source_images(
                settings,
                user_uuid,
                [
                    group_block_uuids[2],
                    group_block_uuids[0],
                    group_block_uuids[2],
                ],
            )
            assert [image.block_uuid for image in images] == [
                group_block_uuids[2],
                group_block_uuids[0],
            ]
            assert [image.data for image in images] == [
                image_bytes[1],
                image_bytes[0],
            ]
            with Image.open(BytesIO(images[0].data)) as image:
                assert image.size == (3, 2)
            assert (
                await application.read_source_images(
                    settings, foreign_user_uuid, [group_block_uuids[0]]
                )
                == []
            )

            assert await graph_counts() == before_counts
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
                    source_uuids=[source.uuid for source in sources],
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User) '
                    'WHERE user.uuid IN $user_uuids '
                    'DETACH DELETE user',
                    user_uuids=[user_uuid, foreign_user_uuid],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_kms_old_source_format_is_rejected_before_reads(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = SourceRetrievalRepository(database.session)
        snapshots = SourceSnapshotRepository(database.session)
        prefix = f'kms-source-format-it-{uuid4().hex}'
        user_uuid = f'{prefix}-user'
        other_user_uuid = f'{prefix}-other-user'
        dimension = VOYAGE_EMBEDDING_DIMENSION
        profile = {'model': 'integration-test', 'dimension': dimension}
        old_image_path = tmp_path / 'old-image.png'
        old_image_bytes = _write_png(old_image_path, 'red')
        old_source = Source(
            uuid=str(uuid4()),
            key='old-format.pdf',
            metadata={'embedding_profile': profile},
        )
        valid_source = Source(
            uuid=str(uuid4()),
            key='valid-format.pdf',
            metadata={'embedding_profile': profile},
        )

        def snapshot(source: Source, *, legacy_image: bool):
            text_block = SourceBlock(
                block_type=BlockType.MARKDOWN,
                content='Text evidence',
                embedding=_embedding(1.0, 0.0, dimension),
            )
            blocks = [text_block]
            if legacy_image:
                blocks.insert(
                    0,
                    SourceBlock(
                        block_type=BlockType.IMAGE,
                        content='Image evidence',
                        embedding=_embedding(1.0, 0.0, dimension),
                        image_path=str(old_image_path),
                        image_sha256=hashlib.sha256(
                            old_image_bytes
                        ).hexdigest(),
                        image_page_indexes=[4],
                    ),
                )
            pages = [SourcePage(index=4, blocks=blocks)]
            member_uuids = [block.uuid for block in blocks]
            instruction = Instruction(
                member_block_uuids=[text_block.uuid],
                governed_statement_uuids=['statement'],
            )
            statement = StatementDraft(
                uuid='statement',
                content='Old-format statement'
                if legacy_image
                else 'Valid statement',
                embedding=_embedding(1.0, 0.0, dimension),
                member_block_uuids=member_uuids,
            )
            procedure = ProcedureDraft(
                content='Old-format procedure'
                if legacy_image
                else 'Valid procedure',
                embedding=_embedding(1.0, 0.0, dimension),
                member_block_uuids=[text_block.uuid],
            )
            return finalize_snapshot_identities(
                source,
                pages,
                [instruction],
                [statement],
                [procedure],
            )

        old_snapshot = snapshot(old_source, legacy_image=True)
        valid_snapshot = snapshot(valid_source, legacy_image=False)
        old_image_uuid = old_snapshot.pages[0].blocks[0].uuid
        old_text_uuid = old_snapshot.pages[0].blocks[1].uuid
        old_statement_uuid = old_snapshot.statements[0].uuid
        old_procedure_uuid = old_snapshot.procedures[0].uuid
        valid_block_uuid = valid_snapshot.pages[0].blocks[0].uuid
        valid_statement_uuid = valid_snapshot.statements[0].uuid
        valid_procedure_uuid = valid_snapshot.procedures[0].uuid
        unknown_uuid = str(uuid4())
        source_uuids = [old_source.uuid, valid_source.uuid]
        user_uuids = [user_uuid, other_user_uuid]

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $user_uuid}), '
                    '(:User {uuid: $other_user_uuid})',
                    user_uuid=user_uuid,
                    other_user_uuid=other_user_uuid,
                )
                await result.consume()
            await snapshots.create_owned_snapshot(user_uuid, old_snapshot)
            await snapshots.create_owned_snapshot(user_uuid, valid_snapshot)
            async with database.session() as session:
                result = await session.run(
                    'MATCH (source:Source {uuid: $source_uuid}) '
                    'REMOVE source.identity_schema '
                    'WITH source '
                    'MATCH (source)-[:HAS_PAGE]->(:SourcePage)'
                    '-[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: $block_uuid}) '
                    'SET block.image_path = $missing_path',
                    source_uuid=old_source.uuid,
                    block_uuid=old_image_uuid,
                    missing_path=str(tmp_path / 'must-not-be-read.png'),
                )
                await result.consume()

            expected_error = 'Source format is obsolete; re-ingest the source'
            with pytest.raises(ValueError, match=f'^{expected_error}$'):
                await application.read_source_passages(
                    settings, user_uuid, [old_text_uuid]
                )
            with pytest.raises(ValueError, match=f'^{expected_error}$'):
                await application.read_source_statements(
                    settings, user_uuid, [old_statement_uuid]
                )
            with pytest.raises(ValueError, match=f'^{expected_error}$'):
                await application.read_source_procedures(
                    settings, user_uuid, [old_procedure_uuid]
                )
            with pytest.raises(ValueError, match=f'^{expected_error}$'):
                await application.read_source_images(
                    settings,
                    user_uuid,
                    [old_image_uuid, old_text_uuid],
                )
            assert (
                await application.read_source_images(
                    settings,
                    user_uuid,
                    [valid_block_uuid],
                )
                == []
            )

            assert (
                await application.read_source_passages(
                    settings, other_user_uuid, [old_text_uuid]
                )
                == []
            )
            assert (
                await application.read_source_passages(
                    settings, user_uuid, [unknown_uuid]
                )
                == []
            )
            with pytest.raises(ResultNotSingleError):
                await application.read_source_statements(
                    settings, other_user_uuid, [old_statement_uuid]
                )
            with pytest.raises(ResultNotSingleError):
                await application.read_source_statements(
                    settings, user_uuid, [unknown_uuid]
                )
            with pytest.raises(ResultNotSingleError):
                await application.read_source_procedures(
                    settings, other_user_uuid, [old_procedure_uuid]
                )
            with pytest.raises(ResultNotSingleError):
                await application.read_source_procedures(
                    settings, user_uuid, [unknown_uuid]
                )
            assert (
                await application.read_source_images(
                    settings, other_user_uuid, [old_image_uuid]
                )
                == []
            )
            assert (
                await application.read_source_images(
                    settings, user_uuid, [unknown_uuid]
                )
                == []
            )

            query_embedding = _embedding(1.0, 0.0, dimension)
            blocks = await repository.search_owned_blocks(
                user_uuid,
                query_embedding,
                candidate_limit=10,
            )
            statements = await repository.search_owned_statements(
                user_uuid,
                query_embedding,
                candidate_limit=10,
            )
            procedures = await repository.search_owned_procedures(
                user_uuid,
                query_embedding,
                candidate_limit=10,
            )
            assert (
                await repository.search_owned_blocks(
                    user_uuid,
                    query_embedding,
                    candidate_limit=10,
                    source_uuids=[old_source.uuid],
                )
                == []
            )
            assert (
                await repository.search_owned_statements(
                    user_uuid,
                    query_embedding,
                    candidate_limit=10,
                    source_uuids=[old_source.uuid],
                )
                == []
            )
            assert (
                await repository.search_owned_procedures(
                    user_uuid,
                    query_embedding,
                    candidate_limit=10,
                    source_uuids=[old_source.uuid],
                )
                == []
            )
            assert [item.block.uuid for item in blocks] == [valid_block_uuid]
            assert [item.statement.uuid for item in statements] == [
                valid_statement_uuid
            ]
            assert [item.procedure.uuid for item in procedures] == [
                valid_procedure_uuid
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
                    'MATCH (user:User) WHERE user.uuid IN $user_uuids '
                    'DETACH DELETE user',
                    user_uuids=user_uuids,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
