"""Prepare and atomically migrate retained research evidence to sources."""

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from neo4j.time import DateTime

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import (
    ResearchSourceMigrationResult,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.database.source.research_source_migration_repository import (
    PreparedResearchSourceMigration,
    ResearchSourceMigrationRepository,
    research_evidence_fingerprint,
)
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)


class ResearchSourceMigrationService:
    """Promote cited historical records without contacting research providers."""

    def __init__(
        self,
        repository: ResearchSourceMigrationRepository,
        snapshot_repository: SourceSnapshotRepository,
        embedding_client: EmbeddingClient,
        storage_root: Path,
    ) -> None:
        self._repository = repository
        self._snapshot_repository = snapshot_repository
        self._embedding_client = embedding_client
        self._storage_root = storage_root

    async def migrate(self, user_uuid: str) -> ResearchSourceMigrationResult:
        """Migrate cited records without synthesizing new content.

        Image-only records remain ordinary image blocks and stay directly
        readable/citable, but are not text-searchable unless retained content
        already contains a description.
        """
        legacy_records = await self._repository.read_cited_evidence(user_uuid)
        if not legacy_records:
            return ResearchSourceMigrationResult(
                migrated_evidence_count=0,
                source_count=0,
            )

        prepared: list[PreparedResearchSourceMigration] = []
        blocks: list[SourceBlock] = []
        for record in legacy_records:
            item, page = await self._prepare_record(user_uuid, record)
            prepared.append(item)
            blocks.extend(page.blocks)

        embedded_blocks = [block for block in blocks if block.content]
        if embedded_blocks:
            vectors = await self._embedding_client.embed(
                [block.content for block in embedded_blocks if block.content]
            )
            for block, vector in zip(embedded_blocks, vectors, strict=True):
                block.embedding = vector

        (
            migrated_count,
            source_count,
        ) = await self._repository.migrate_cited_evidence(
            user_uuid,
            prepared,
        )
        return ResearchSourceMigrationResult(
            migrated_evidence_count=migrated_count,
            source_count=source_count,
        )

    async def _prepare_record(
        self,
        user_uuid: str,
        record: dict[str, Any],
    ) -> tuple[PreparedResearchSourceMigration, SourcePage]:
        media_path = await self._copy_capture(
            record['media_path'],
            record['media_type'],
        )
        original_media_path = await self._copy_capture(
            record['original_media_path'],
            record['original_media_type'],
        )
        evidence_uuid = record['uuid']
        raw_content = record['raw_content']
        page_markdown = raw_content or ''
        content = record['content']
        page_index = record['page_index']
        kind = record['kind']
        url = record['url']
        resolved_url = record['resolved_url']
        image_bytes = (
            await asyncio.to_thread(Path(media_path).read_bytes)
            if media_path is not None
            else None
        )
        original_bytes = (
            await asyncio.to_thread(Path(original_media_path).read_bytes)
            if original_media_path is not None
            else None
        )
        digest_payload = {
            'kind': kind,
            'url': url,
            'resolved_url': resolved_url,
            'page_index': page_index,
            'line_start': record['line_start'],
            'line_end': record['line_end'],
            'raw_content': raw_content,
            'content': content,
            'media_sha256': _sha256(image_bytes),
            'original_media_sha256': _sha256(original_bytes),
        }
        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()
        source_uuid = str(
            uuid5(
                NAMESPACE_URL,
                f'kms2-legacy-research:{user_uuid}:{evidence_uuid}:{digest}',
            )
        )
        retrieved_at = _native_datetime(record['retrieved_at'])
        scope = {
            'web_passage': 'retained_fragment',
            'image': 'retained_fragment',
            'pdf_page': 'selected_pdf_pages',
        }[kind]
        title = record['title']
        metadata: dict[str, object] = {
            'origin': 'legacy_research_snapshot',
            'capture_scope': scope,
            'url': url,
            'resolved_url': resolved_url,
            'title': title,
            'retrieved_at': retrieved_at.isoformat(),
            'snapshot_digest': digest,
            'legacy_raw_content': raw_content,
            'line_start': record['line_start'],
            'line_end': record['line_end'],
            'selected_page_indexes': [page_index] if kind == 'pdf_page' else [],
            'media_capture_path': media_path,
            'media_type': record['media_type'],
            'original_media_capture_path': original_media_path,
            'original_media_type': record['original_media_type'],
        }
        source = Source(
            uuid=source_uuid,
            key=title or url,
            metadata=metadata,
        )
        block_uuid = str(uuid5(NAMESPACE_URL, f'{source_uuid}:block:0'))
        asset: list[VisualAsset] = []
        if media_path is not None:
            asset.append(
                VisualAsset(
                    uuid=str(uuid5(NAMESPACE_URL, f'{source_uuid}:asset:0')),
                    path=media_path,
                )
            )
        block_type = (
            BlockType.IMAGE
            if kind == 'image' or kind == 'pdf_page' and not content
            else BlockType.MARKDOWN
        )
        block = SourceBlock(
            uuid=block_uuid,
            block_type=block_type,
            content=content,
            assets=asset,
        )
        page = SourcePage(
            index=page_index if kind == 'pdf_page' else 0,
            markdown=page_markdown,
            blocks=[block],
        )
        existing = await self._snapshot_repository.load_owned_snapshot(
            user_uuid,
            source_uuid,
        )
        expected_revision = (
            existing.source.revision if existing is not None else 0
        )
        return (
            PreparedResearchSourceMigration(
                evidence_uuid=evidence_uuid,
                source=source,
                pages=[page],
                expected_revision=expected_revision,
                legacy_fingerprint=research_evidence_fingerprint(record),
                citation_kind=kind,
            ),
            page,
        )

    async def _copy_capture(
        self,
        path: str | None,
        media_type: str | None,
    ) -> str | None:
        if path is None:
            return None
        data = await asyncio.to_thread(Path(path).read_bytes)
        destination = await asyncio.to_thread(
            self._store_content_addressed,
            data,
            _media_suffix(media_type, path),
        )
        return str(destination)

    def _store_content_addressed(self, data: bytes, suffix: str) -> Path:
        """Copy one legacy blob to durable content-addressed storage."""
        digest = hashlib.sha256(data).hexdigest()
        self._storage_root.mkdir(parents=True, exist_ok=True)
        destination = self._storage_root / f'{digest}{suffix}'
        if destination.exists():
            return destination

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f'.{digest}.',
            dir=self._storage_root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as temporary_file:
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination


class ResearchSourceMigrationFinalizationService:
    """Expose global legacy cleanup as a database-only workflow."""

    def __init__(
        self,
        repository: ResearchSourceMigrationRepository,
    ) -> None:
        self._repository = repository

    async def finalize(self) -> None:
        """Remove orphaned legacy snapshots and drop their uniqueness rule."""
        await self._repository.drop_external_evidence_constraint()


def _native_datetime(value: datetime | DateTime) -> datetime:
    """Convert Neo4j temporal values to standard-library datetimes."""
    return value.to_native() if isinstance(value, DateTime) else value


def _sha256(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _media_suffix(media_type: str | None, path: str) -> str:
    """Retain a recognized image extension for file-backed image readers."""
    if media_type == 'image/png':
        return '.png'
    if media_type == 'image/jpeg':
        return '.jpg'
    if media_type == 'image/webp':
        return '.webp'
    if media_type == 'image/svg+xml':
        return '.svg'
    suffix = Path(path).suffix.lower()
    return (
        suffix if suffix in {'.png', '.jpg', '.jpeg', '.svg', '.webp'} else ''
    )
