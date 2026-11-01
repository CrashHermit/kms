"""Behavioral checks for explicit historical research-source migration."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from kms2.core.model.source_processing.admission import (
    ResearchSourceMigrationResult,
)
from kms2.database.source.research_source_migration_repository import (
    research_evidence_fingerprint,
)
from kms2.service.research_source_migration import (
    ResearchSourceMigrationFinalizationService,
    ResearchSourceMigrationService,
)


class _EmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


class _MigrationRepository:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.prepared = []

    async def read_cited_evidence(self, user_uuid: str):
        return self.records

    async def migrate_cited_evidence(self, user_uuid: str, prepared):
        self.prepared = prepared
        return len(prepared), len({item.source.uuid for item in prepared})


class _SnapshotRepository:
    async def load_owned_snapshot(self, user_uuid: str, source_uuid: str):
        return None


def _record(
    *,
    uuid: str,
    kind: str,
    url: str,
    raw_content: str,
    content: str | None,
    page_index: int | None = None,
    media_path: str | None = None,
    media_type: str | None = None,
    original_media_path: str | None = None,
    original_media_type: str | None = None,
) -> dict[str, object]:
    return {
        'uuid': uuid,
        'kind': kind,
        'url': url,
        'resolved_url': None,
        'title': f'Title {uuid}',
        'retrieved_at': datetime(2026, 1, 1, tzinfo=UTC),
        'page_index': page_index,
        'line_start': 3,
        'line_end': 8,
        'raw_content': raw_content,
        'content': content,
        'media_path': media_path,
        'media_type': media_type,
        'original_media_path': original_media_path,
        'original_media_type': original_media_type,
        'turn_uuids': [f'turn-{uuid}'],
        'session_uuids': [f'session-{uuid}'],
    }


def test_legacy_fingerprint_tracks_media_metadata_and_retained_content() -> (
    None
):
    record = _record(
        uuid='legacy-image',
        kind='image',
        url='https://example.org/figure.png',
        raw_content='Original caption.',
        content='Retained description.',
        media_path='/capture/inspected.png',
        media_type='image/png',
        original_media_path='/capture/original.svg',
        original_media_type='image/svg+xml',
    )
    fingerprint = research_evidence_fingerprint(record)

    for field, value in (
        ('raw_content', 'Changed raw caption.'),
        ('content', 'Changed description.'),
        ('media_path', '/capture/other.png'),
        ('media_type', 'image/webp'),
        ('original_media_path', '/capture/other.svg'),
        ('original_media_type', 'image/jpeg'),
    ):
        changed = {**record, field: value}
        assert research_evidence_fingerprint(changed) != fingerprint


def test_migration_preserves_historical_content_and_media_as_owned_sources(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        inspected_png = tmp_path / 'inspected.png'
        inspected_png.write_bytes(b'inspected PNG bytes')
        original_svg = tmp_path / 'original.svg'
        original_svg.write_bytes(b'<svg>original</svg>')
        page_render = tmp_path / 'page.png'
        page_render.write_bytes(b'page render bytes')
        records = [
            _record(
                uuid='legacy-web',
                kind='web_passage',
                url='https://example.org/page?x=1#source',
                raw_content='The original web fragment.',
                content='Formatted web fragment.',
            ),
            _record(
                uuid='legacy-image',
                kind='image',
                url='https://example.org/figure.svg#figure',
                raw_content='Figure caption.',
                content='A diagram of a mapping.',
                media_path=str(inspected_png),
                media_type='image/png',
                original_media_path=str(original_svg),
                original_media_type='image/svg+xml',
            ),
            _record(
                uuid='legacy-image-only',
                kind='image',
                url='https://example.org/no-description.png',
                raw_content='',
                content=None,
                media_path=str(inspected_png),
                media_type='image/png',
            ),
            _record(
                uuid='legacy-pdf',
                kind='pdf_page',
                url='https://example.org/solution.pdf',
                raw_content='Original page text.',
                content='Formatted page text.',
                page_index=12,
                media_path=str(page_render),
                media_type='image/png',
            ),
        ]
        records[2]['raw_content'] = None
        repository = _MigrationRepository(records)
        embedding_client = _EmbeddingClient()
        storage_root = tmp_path / 'durable'
        service = ResearchSourceMigrationService(
            repository,  # type: ignore[arg-type]
            _SnapshotRepository(),  # type: ignore[arg-type]
            embedding_client,
            storage_root,
        )

        result = await service.migrate('owner-1')

        assert result == ResearchSourceMigrationResult(
            migrated_evidence_count=4,
            source_count=4,
        )
        assert embedding_client.texts == [
            'Formatted web fragment.',
            'A diagram of a mapping.',
            'Formatted page text.',
        ]
        by_legacy_id = {
            item.evidence_uuid: item for item in repository.prepared
        }
        web = by_legacy_id['legacy-web']
        assert web.source.key == 'Title legacy-web'
        assert (
            web.source.metadata['url'] == 'https://example.org/page?x=1#source'
        )
        assert web.source.metadata['capture_scope'] == 'retained_fragment'
        assert web.source.metadata['legacy_raw_content'] == (
            'The original web fragment.'
        )
        assert web.pages[0].markdown == 'The original web fragment.'
        assert web.pages[0].blocks[0].content == 'Formatted web fragment.'
        assert web.pages[0].blocks[0].embedding == [1.0, 0.0]

        image = by_legacy_id['legacy-image']
        image_only = by_legacy_id['legacy-image-only']
        assert image_only.source.metadata['legacy_raw_content'] is None
        assert image_only.pages[0].markdown == ''
        assert image_only.pages[0].blocks[0].content is None
        assert image_only.pages[0].blocks[0].embedding is None
        assert image_only.pages[0].blocks[0].assets[0].path.endswith('.png')
        asset_path = Path(image.pages[0].blocks[0].assets[0].path)
        original_path = Path(
            image.source.metadata['original_media_capture_path']
        )
        assert asset_path.suffix == '.png'
        assert asset_path.read_bytes() == b'inspected PNG bytes'
        assert original_path.suffix == '.svg'
        assert original_path.read_bytes() == b'<svg>original</svg>'
        assert image.source.metadata['media_type'] == 'image/png'
        assert image.source.metadata['original_media_type'] == 'image/svg+xml'
        assert image.pages[0].blocks[0].embedding == [2.0, 0.0]

        pdf = by_legacy_id['legacy-pdf']
        assert pdf.source.metadata['capture_scope'] == 'selected_pdf_pages'
        assert pdf.source.metadata['selected_page_indexes'] == [12]
        assert pdf.pages[0].index == 12
        assert pdf.pages[0].markdown == 'Original page text.'
        assert Path(pdf.pages[0].blocks[0].assets[0].path).read_bytes() == (
            b'page render bytes'
        )
        assert all(path.is_file() for path in storage_root.iterdir())

    asyncio.run(exercise())


def test_empty_migration_is_a_no_op(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = _MigrationRepository([])
        embedding_client = _EmbeddingClient()
        storage_root = tmp_path / 'durable'
        service = ResearchSourceMigrationService(
            repository,  # type: ignore[arg-type]
            _SnapshotRepository(),  # type: ignore[arg-type]
            embedding_client,
            storage_root,
        )

        result = await service.migrate('owner-without-legacy-evidence')

        assert result == ResearchSourceMigrationResult(
            migrated_evidence_count=0,
            source_count=0,
        )
        assert repository.prepared == []
        assert embedding_client.texts == []
        assert not storage_root.exists()

    asyncio.run(exercise())


class _FinalizationRepository:
    def __init__(self) -> None:
        self.finalized = False

    async def drop_external_evidence_constraint(self) -> None:
        self.finalized = True


def test_finalization_service_delegates_to_repository() -> None:
    async def exercise() -> None:
        repository = _FinalizationRepository()
        service = ResearchSourceMigrationFinalizationService(
            repository,  # type: ignore[arg-type]
        )

        await service.finalize()

        assert repository.finalized

    asyncio.run(exercise())
