"""Opt-in Neo4j checks for atomic historical research-source migration."""

import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError
from PIL import Image

from kms2.composition.embedding import embedding_dimension
from kms2.config.services import DatabaseSettings
from kms2.config.settings import Settings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.research_source_migration_repository import (
    PreparedResearchSourceMigration,
    ResearchSourceMigrationRepository,
    research_evidence_fingerprint,
)
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms2.service.research_source_migration import (
    ResearchSourceMigrationService,
)

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


class _EmbeddingClient:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * (self.dimension - 1))] for _ in texts]


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database=DatabaseSettings(
            uri=os.environ['KMS2_DATABASE__URI'],
            username=os.environ['KMS2_DATABASE__USERNAME'],
            password=os.environ['KMS2_DATABASE__PASSWORD'],
            database=os.environ['KMS2_DATABASE__DATABASE'],
        ),
    )


def _png(color: str) -> bytes:
    image = Image.new('RGB', (4, 3), color)
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_research_source_migration_preserves_history_and_shared_legacy_records(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        settings = _settings()
        database = DatabaseClient(settings.database)
        repository = ResearchSourceMigrationRepository(database.session)
        snapshots = SourceSnapshotRepository(database.session)
        prefix = f'kms2-research-migration-it-{uuid4().hex}'
        users = [f'{prefix}-user-{index}' for index in range(2)]
        turns = {
            user_uuid: [
                f'{prefix}-turn-{user_index}-{index}' for index in range(3)
            ]
            for user_index, user_uuid in enumerate(users)
        }
        sessions = {
            user_uuid: f'{prefix}-session-{index}'
            for index, user_uuid in enumerate(users)
        }
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        png_path = tmp_path / 'inspected.png'
        png_path.write_bytes(_png('blue'))
        pdf_path = tmp_path / 'page-render.png'
        pdf_path.write_bytes(_png('green'))
        original_path = tmp_path / 'original.svg'
        original_bytes = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        original_path.write_bytes(original_bytes)
        evidence = [
            {
                'uuid': f'{prefix}-web',
                'kind': 'web_passage',
                'url': 'https://example.org/article#historical',
                'resolved_url': 'https://example.org/article#historical',
                'title': f'{prefix} web passage',
                'retrieved_at': created_at,
                'page_index': None,
                'line_start': 4,
                'line_end': 8,
                'raw_content': 'Original retained web fragment.',
                'content': 'Formatted retained web fragment.',
                'media_path': None,
                'media_type': None,
                'original_media_path': None,
                'original_media_type': None,
            },
            {
                'uuid': f'{prefix}-image',
                'kind': 'image',
                'url': 'https://example.org/figure.svg',
                'resolved_url': 'https://cdn.example.org/figure.svg',
                'title': f'{prefix} image',
                'retrieved_at': created_at,
                'page_index': None,
                'line_start': None,
                'line_end': None,
                'raw_content': 'Retained caption.',
                'content': 'The figure shows a linear map.',
                'media_path': str(png_path),
                'media_type': 'image/png',
                'original_media_path': str(original_path),
                'original_media_type': 'image/svg+xml',
            },
            {
                'uuid': f'{prefix}-pdf',
                'kind': 'pdf_page',
                'url': 'https://example.org/solution.pdf',
                'resolved_url': 'https://example.org/solution.pdf',
                'title': f'{prefix} selected PDF page',
                'retrieved_at': created_at,
                'page_index': 17,
                'line_start': None,
                'line_end': None,
                'raw_content': 'Original retained PDF page text.',
                'content': 'Formatted retained PDF page text.',
                'media_path': str(pdf_path),
                'media_type': 'image/png',
                'original_media_path': None,
                'original_media_type': None,
            },
        ]
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=embedding_dimension(settings),
            )
            citation_rows = [
                {
                    'user_uuid': user_uuid,
                    'session_uuid': sessions[user_uuid],
                    'turn_uuid': turns[user_uuid][evidence_index],
                    'evidence_uuid': record['uuid'],
                }
                for user_uuid in users
                for evidence_index, record in enumerate(evidence)
            ]
            async with database.session() as session:
                async with await session.begin_transaction() as transaction:
                    result = await transaction.run(
                        """
                        UNWIND $users AS user_uuid
                        CREATE (:User {uuid: user_uuid})
                        """,
                        users=users,
                    )
                    await result.consume()
                    result = await transaction.run(
                        """
                        UNWIND $records AS record
                        CREATE (evidence:ExternalEvidence)
                        SET evidence = record
                        """,
                        records=evidence,
                    )
                    await result.consume()
                    result = await transaction.run(
                        """
                        UNWIND $citations AS citation
                        MATCH (user:User {uuid: citation.user_uuid})
                        MERGE (session:TeachingSession {
                            uuid: citation.session_uuid
                        })
                        MERGE (user)-[:HAS_TEACHING_SESSION]->(session)
                        CREATE (turn:TeachingTurn {
                            uuid: citation.turn_uuid,
                            role: 'teacher',
                            content: 'Historical response',
                            created_at: $created_at
                        })
                        CREATE (session)-[:HAS_TURN]->(turn)
                        WITH turn, citation
                        MATCH (evidence:ExternalEvidence {
                            uuid: citation.evidence_uuid
                        })
                        CREATE (turn)-[:CITES_EXTERNAL]->(evidence)
                        """,
                        citations=citation_rows,
                        created_at=created_at,
                    )
                    await result.consume()

            service = ResearchSourceMigrationService(
                repository,
                snapshots,
                _EmbeddingClient(embedding_dimension(settings)),
                tmp_path / 'durable-sources',
            )
            with pytest.raises(ValueError, match='CITES_EXTERNAL'):
                await repository.drop_external_evidence_constraint()

            first = await service.migrate(users[0])
            assert first.migrated_evidence_count == 3
            assert first.source_count == 3
            assert await _count_external(database, users[0], prefix) == (0, 3)
            assert await _citation_counts(database, users[0]) == (2, 2)

            owned_source_rows = await _owned_sources(database, users[0])
            assert len(owned_source_rows) == 3
            source_by_url = {row['url']: row for row in owned_source_rows}
            web_snapshot = await snapshots.load_owned_snapshot(
                users[0],
                source_by_url['https://example.org/article#historical']['uuid'],
            )
            assert web_snapshot is not None
            assert web_snapshot.pages[0].index == 0
            assert web_snapshot.pages[0].markdown == (
                'Original retained web fragment.'
            )
            assert web_snapshot.pages[0].blocks[0].content == (
                'Formatted retained web fragment.'
            )
            assert web_snapshot.source.metadata['capture_scope'] == (
                'retained_fragment'
            )

            image_snapshot = await snapshots.load_owned_snapshot(
                users[0],
                source_by_url['https://example.org/figure.svg']['uuid'],
            )
            assert image_snapshot is not None
            migrated_png = Path(
                image_snapshot.pages[0].blocks[0].assets[0].path
            )
            migrated_svg = Path(
                image_snapshot.source.metadata['original_media_capture_path']
            )
            assert migrated_png.read_bytes() == png_path.read_bytes()
            assert migrated_svg.read_bytes() == original_bytes
            assert image_snapshot.pages[0].blocks[0].content == (
                'The figure shows a linear map.'
            )

            pdf_snapshot = await snapshots.load_owned_snapshot(
                users[0],
                source_by_url['https://example.org/solution.pdf']['uuid'],
            )
            assert pdf_snapshot is not None
            assert pdf_snapshot.pages[0].index == 17
            assert pdf_snapshot.pages[0].markdown == (
                'Original retained PDF page text.'
            )
            assert (
                pdf_snapshot.pages[0].blocks[0].assets[0].path.endswith('.png')
            )

            second = await service.migrate(users[1])
            assert second.migrated_evidence_count == 3
            assert second.source_count == 3
            assert await _count_external(database, users[1], prefix) == (0, 0)
            assert await _citation_counts(database, users[1]) == (2, 2)
            assert (
                await service.migrate(users[0])
            ).migrated_evidence_count == 0
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:ExternalEvidence {uuid: $uuid})',
                    uuid=f'{prefix}-orphan',
                )
                await result.consume()
            if await _global_external_citation_count(database):
                with pytest.raises(ValueError, match='CITES_EXTERNAL'):
                    await repository.drop_external_evidence_constraint()
                assert await _count_external(database, users[0], prefix) == (
                    0,
                    1,
                )
            else:
                await repository.drop_external_evidence_constraint()
                assert await _count_external(database, users[0], prefix) == (
                    0,
                    0,
                )
                await repository.drop_external_evidence_constraint()
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH $prefix
                    DETACH DELETE node
                    """,
                    prefix=prefix,
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.key STARTS WITH $prefix
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->
                                  (asset:VisualAsset)
                    WITH collect(DISTINCT source) AS sources,
                         collect(DISTINCT page) AS pages,
                         collect(DISTINCT block) AS blocks,
                         collect(DISTINCT asset) AS assets
                    FOREACH (asset IN assets | DETACH DELETE asset)
                    FOREACH (block IN blocks | DETACH DELETE block)
                    FOREACH (page IN pages | DETACH DELETE page)
                    FOREACH (source IN sources | DETACH DELETE source)
                    """,
                    prefix=prefix,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_research_migration_rejects_stale_source_revision(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        settings = _settings()
        database = DatabaseClient(settings.database)
        repository = ResearchSourceMigrationRepository(database.session)
        snapshots = SourceSnapshotRepository(database.session)
        prefix = f'kms2-research-migration-stale-it-{uuid4().hex}'
        user_uuid = f'{prefix}-user'
        session_uuid = f'{prefix}-session'
        turn_uuid = f'{prefix}-turn'
        source = Source(
            uuid=f'{prefix}-source',
            key=f'{prefix} source',
            metadata={'snapshot_digest': 'captured-input'},
        )
        block = SourceBlock(
            uuid=f'{prefix}-block',
            block_type=BlockType.MARKDOWN,
            content='Retained historical text.',
            embedding=[
                1.0,
                *([0.0] * (embedding_dimension(settings) - 1)),
            ],
        )
        pages = [
            SourcePage(
                index=0,
                markdown='Retained historical text.',
                blocks=[block],
            )
        ]
        record = {
            'uuid': f'{prefix}-legacy',
            'kind': 'web_passage',
            'url': 'https://example.org/retained',
            'resolved_url': None,
            'title': f'{prefix} historical fragment',
            'retrieved_at': datetime(2026, 1, 1, tzinfo=UTC),
            'page_index': None,
            'line_start': None,
            'line_end': None,
            'raw_content': 'Retained historical text.',
            'content': 'Retained historical text.',
            'media_path': None,
            'media_type': None,
            'original_media_path': None,
            'original_media_type': None,
        }
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=embedding_dimension(settings),
            )
            async with database.session() as session:
                async with await session.begin_transaction() as transaction:
                    result = await transaction.run(
                        'CREATE (:User {uuid: $uuid})',
                        uuid=user_uuid,
                    )
                    await result.consume()
                    result = await transaction.run(
                        """
                        CREATE (session:TeachingSession {uuid: $session_uuid})
                        CREATE (turn:TeachingTurn {
                            uuid: $turn_uuid,
                            role: 'teacher',
                            content: 'Historical response',
                            created_at: $created_at
                        })
                        CREATE (evidence:ExternalEvidence)
                        SET evidence = $evidence
                        CREATE (session)-[:HAS_TURN]->(turn)
                        CREATE (turn)-[:CITES_EXTERNAL]->(evidence)
                        WITH session
                        MATCH (user:User {uuid: $user_uuid})
                        CREATE (user)-[:HAS_TEACHING_SESSION]->(session)
                        """,
                        session_uuid=session_uuid,
                        turn_uuid=turn_uuid,
                        created_at=record['retrieved_at'],
                        evidence=record,
                        user_uuid=user_uuid,
                    )
                    await result.consume()
            await snapshots.create_owned_snapshot(user_uuid, source, pages)
            async with database.session() as session:
                result = await session.run(
                    'MATCH (source:Source {uuid: $uuid}) '
                    'SET source.revision = 1',
                    uuid=source.uuid,
                )
                await result.consume()

            prepared = PreparedResearchSourceMigration(
                evidence_uuid=record['uuid'],
                source=source,
                pages=pages,
                expected_revision=0,
                legacy_fingerprint=research_evidence_fingerprint(record),
                citation_kind='web_passage',
            )
            with pytest.raises(ResultNotSingleError):
                await repository.migrate_cited_evidence(user_uuid, [prepared])
            stale_legacy = replace(
                prepared,
                expected_revision=1,
                legacy_fingerprint='changed-while-preparing',
            )
            with pytest.raises(
                ResultNotSingleError,
                match='changed after preparation',
            ):
                await repository.migrate_cited_evidence(
                    user_uuid,
                    [stale_legacy],
                )
            assert await _count_external(database, user_uuid, prefix) == (1, 1)
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->
                                  (asset:VisualAsset)
                    WITH collect(DISTINCT source) AS sources,
                         collect(DISTINCT page) AS pages,
                         collect(DISTINCT block) AS blocks,
                         collect(DISTINCT asset) AS assets
                    FOREACH (asset IN assets | DETACH DELETE asset)
                    FOREACH (block IN blocks | DETACH DELETE block)
                    FOREACH (page IN pages | DETACH DELETE page)
                    FOREACH (source IN sources | DETACH DELETE source)
                    """,
                    source_uuid=source.uuid,
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH $prefix
                    DETACH DELETE node
                    """,
                    prefix=prefix,
                )
                await result.consume()
            await database.close()


async def _count_external(
    database: DatabaseClient,
    user_uuid: str,
    prefix: str,
) -> tuple[int, int]:
    async with database.session() as session:
        result = await session.run(
            """
            MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
                  (:TeachingSession)-[:HAS_TURN]->(turn:TeachingTurn)
            OPTIONAL MATCH (turn)-[:CITES_EXTERNAL]->(evidence:ExternalEvidence)
            WITH count(DISTINCT evidence) AS cited
            CALL {
                MATCH (legacy:ExternalEvidence)
                WHERE legacy.uuid STARTS WITH $prefix
                RETURN count(legacy) AS total
            }
            RETURN cited, total
            """,
            user_uuid=user_uuid,
            prefix=prefix,
        )
        row = await result.single(strict=True)
        return row['cited'], row['total']


async def _global_external_citation_count(database: DatabaseClient) -> int:
    async with database.session() as session:
        result = await session.run(
            'MATCH ()-[citation:CITES_EXTERNAL]->() '
            'RETURN count(citation) AS count'
        )
        row = await result.single(strict=True)
        return row['count']


async def _citation_counts(
    database: DatabaseClient, user_uuid: str
) -> tuple[int, int]:
    async with database.session() as session:
        result = await session.run(
            """
            MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
                  (:TeachingSession)-[:HAS_TURN]->(turn:TeachingTurn)
            OPTIONAL MATCH (turn)-[:CITES_BLOCK]->(block:SourceBlock)
            OPTIONAL MATCH (turn)-[:CITES_IMAGE]->(image:VisualAsset)
            RETURN count(DISTINCT block) AS blocks,
                   count(DISTINCT image) AS images
            """,
            user_uuid=user_uuid,
        )
        row = await result.single(strict=True)
        return row['blocks'], row['images']


async def _owned_sources(database: DatabaseClient, user_uuid: str):
    async with database.session() as session:
        result = await session.run(
            """
            MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
            RETURN source.uuid AS uuid,
                   source.key AS key,
                   source.metadata AS metadata
            ORDER BY source.key
            """,
            user_uuid=user_uuid,
        )
        rows = await result.data()
        for row in rows:
            row['metadata'] = json.loads(row['metadata'])
            row['url'] = row['metadata']['url']
        return rows
