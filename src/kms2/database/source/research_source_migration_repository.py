"""Atomic persistence for converting legacy research citations to sources."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from neo4j import AsyncSession, AsyncTransaction, Record
from neo4j.exceptions import ResultNotSingleError
from neo4j.time import DateTime

from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import OwnedSourceSnapshot
from kms2.database.source.queries.research_source_migration import (
    CHECK_RESEARCH_MIGRATION_SOURCE_REVISIONS,
    COUNT_LEGACY_RESEARCH_CITATIONS,
    COUNT_LEGACY_RESEARCH_EVIDENCE,
    COUNT_REMAINING_USER_RESEARCH_CITATIONS,
    DELETE_ORPHANED_LEGACY_RESEARCH_EVIDENCE,
    DELETE_UNREFERENCED_LEGACY_RESEARCH_EVIDENCE,
    DROP_EXTERNAL_EVIDENCE_CONSTRAINT,
    LOCK_LEGACY_RESEARCH_EVIDENCE,
    LOCK_RESEARCH_MIGRATION_SESSIONS,
    LOCK_RESEARCH_MIGRATION_SOURCES,
    LOCK_RESEARCH_MIGRATION_USER,
    READ_USER_RESEARCH_CITATIONS,
    REWRITE_RESEARCH_CITATIONS,
)
from kms2.database.source.source_snapshot_repository import (
    create_owned_snapshot_in_transaction,
)


@dataclass(frozen=True)
class PreparedResearchSourceMigration:
    """One evidence snapshot prepared before its database transaction."""

    evidence_uuid: str
    source: Source
    pages: list[SourcePage]
    expected_revision: int
    legacy_fingerprint: str
    citation_kind: str


class ResearchSourceMigrationRepository:
    """Migrate one owner's cited legacy evidence in a single transaction."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def drop_external_evidence_constraint(self) -> None:
        """Clean legacy evidence before dropping its obsolete uniqueness rule."""
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                citations = await self._single(
                    transaction,
                    COUNT_LEGACY_RESEARCH_CITATIONS,
                )
                if citations['count']:
                    raise ValueError(
                        'CITES_EXTERNAL relationships remain; migrate every '
                        'owner before dropping the constraint'
                    )
                result = await transaction.run(
                    DELETE_ORPHANED_LEGACY_RESEARCH_EVIDENCE
                )
                await result.consume()
                legacy = await self._single(
                    transaction,
                    COUNT_LEGACY_RESEARCH_EVIDENCE,
                )
                if legacy['count']:
                    raise ResultNotSingleError(
                        'Legacy external evidence remains after global cleanup'
                    )

            async with await session.begin_transaction() as transaction:
                citations = await self._single(
                    transaction,
                    COUNT_LEGACY_RESEARCH_CITATIONS,
                )
                if citations['count']:
                    raise ValueError(
                        'CITES_EXTERNAL relationships remain; migrate every '
                        'owner before dropping the constraint'
                    )
                legacy = await self._single(
                    transaction,
                    COUNT_LEGACY_RESEARCH_EVIDENCE,
                )
                if legacy['count']:
                    raise ResultNotSingleError(
                        'Legacy external evidence remains after global cleanup'
                    )

        async with self._session_factory() as session:
            result = await session.run(DROP_EXTERNAL_EVIDENCE_CONSTRAINT)
            await result.consume()

    async def read_cited_evidence(
        self,
        user_uuid: str,
    ) -> list[dict[str, object]]:
        """Read distinct cited legacy records and this owner's citing turns."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_USER_RESEARCH_CITATIONS,
                user_uuid=user_uuid,
            )
            return await result.data()

    async def migrate_cited_evidence(
        self,
        user_uuid: str,
        prepared: list[PreparedResearchSourceMigration],
    ) -> tuple[int, int]:
        """Create/reuse snapshots, rewrite citations, and collect old records."""
        if not prepared:
            return 0, 0

        prepared_by_uuid = {item.evidence_uuid: item for item in prepared}
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._single(
                    transaction,
                    LOCK_RESEARCH_MIGRATION_USER,
                    user_uuid=user_uuid,
                )
                result = await transaction.run(
                    READ_USER_RESEARCH_CITATIONS,
                    user_uuid=user_uuid,
                )
                current_rows = await result.data()
                current_by_uuid = {
                    row['uuid']: row
                    for row in current_rows
                    if row['uuid'] in prepared_by_uuid
                }
                if not current_by_uuid:
                    return 0, 0

                active = [
                    prepared_by_uuid[evidence_uuid]
                    for evidence_uuid in sorted(current_by_uuid)
                ]
                evidence_uuids = [item.evidence_uuid for item in active]
                source_uuids = sorted({item.source.uuid for item in active})
                locked_evidence = await self._single(
                    transaction,
                    LOCK_LEGACY_RESEARCH_EVIDENCE,
                    evidence_uuids=evidence_uuids,
                )
                if set(locked_evidence['uuids']) != set(evidence_uuids):
                    raise ResultNotSingleError(
                        'Legacy research evidence changed during migration'
                    )
                result = await transaction.run(
                    READ_USER_RESEARCH_CITATIONS,
                    user_uuid=user_uuid,
                )
                locked_rows = await result.data()
                locked_by_uuid = {
                    row['uuid']: row
                    for row in locked_rows
                    if row['uuid'] in prepared_by_uuid
                }
                if set(locked_by_uuid) != set(evidence_uuids):
                    raise ResultNotSingleError(
                        'Legacy research evidence changed during migration'
                    )
                for item in active:
                    if (
                        research_evidence_fingerprint(
                            locked_by_uuid[item.evidence_uuid]
                        )
                        != item.legacy_fingerprint
                    ):
                        raise ResultNotSingleError(
                            'Legacy research evidence changed after preparation'
                        )
                current_by_uuid = locked_by_uuid
                session_uuids = sorted(
                    {
                        session_uuid
                        for row in current_by_uuid.values()
                        for session_uuid in row['session_uuids']
                    }
                )
                await self._single(
                    transaction,
                    LOCK_RESEARCH_MIGRATION_SOURCES,
                    source_uuids=source_uuids,
                )

                snapshots: dict[str, OwnedSourceSnapshot] = {}
                for item in active:
                    snapshots[
                        item.evidence_uuid
                    ] = await create_owned_snapshot_in_transaction(
                        transaction,
                        user_uuid,
                        item.source,
                        item.pages,
                    )

                await self._single(
                    transaction,
                    LOCK_RESEARCH_MIGRATION_SOURCES,
                    source_uuids=source_uuids,
                )
                expected_sources = [
                    {
                        'uuid': snapshots[item.evidence_uuid].source.uuid,
                        'revision': item.expected_revision,
                    }
                    for item in active
                ]
                source_tokens = await self._single(
                    transaction,
                    CHECK_RESEARCH_MIGRATION_SOURCE_REVISIONS,
                    sources=expected_sources,
                )
                if set(source_tokens['uuids']) != set(source_uuids):
                    raise ResultNotSingleError(
                        'Research source changed before its citations migrated'
                    )
                locked_sessions = await self._single(
                    transaction,
                    LOCK_RESEARCH_MIGRATION_SESSIONS,
                    user_uuid=user_uuid,
                    session_uuids=session_uuids,
                )
                if set(locked_sessions['uuids']) != set(session_uuids):
                    raise ResultNotSingleError(
                        'Teaching session changed during source migration'
                    )

                rewrites = [
                    self._rewrite(
                        item,
                        snapshots[item.evidence_uuid],
                        turn_uuid,
                    )
                    for item in active
                    for turn_uuid in sorted(
                        current_by_uuid[item.evidence_uuid]['turn_uuids']
                    )
                ]
                migrated = await self._single(
                    transaction,
                    REWRITE_RESEARCH_CITATIONS,
                    user_uuid=user_uuid,
                    rewrites=rewrites,
                )
                if migrated['migrated'] != len(rewrites):
                    raise ResultNotSingleError(
                        'Research citation changed before its source migration'
                    )
                await self._single(
                    transaction,
                    DELETE_UNREFERENCED_LEGACY_RESEARCH_EVIDENCE,
                    evidence_uuids=evidence_uuids,
                )
                remaining = await self._single(
                    transaction,
                    COUNT_REMAINING_USER_RESEARCH_CITATIONS,
                    user_uuid=user_uuid,
                )
                if remaining['count']:
                    raise ResultNotSingleError(
                        'Legacy research citations remain after migration'
                    )

        return len(evidence_uuids), len(source_uuids)

    @staticmethod
    def _rewrite(
        item: PreparedResearchSourceMigration,
        snapshot: OwnedSourceSnapshot,
        turn_uuid: str,
    ) -> dict[str, str | None]:
        page = snapshot.pages[0]
        block = page.blocks[0]
        asset_uuid = block.assets[0].uuid if block.assets else None
        return {
            'evidence_uuid': item.evidence_uuid,
            'turn_uuid': turn_uuid,
            'source_uuid': snapshot.source.uuid,
            'block_uuid': block.uuid,
            'asset_uuid': asset_uuid,
            'kind': item.citation_kind,
        }

    @staticmethod
    async def _single(
        transaction: AsyncTransaction,
        query: str,
        **parameters: object,
    ) -> Record:
        result = await transaction.run(query, **parameters)
        return await result.single(strict=True)


def research_evidence_fingerprint(record: Mapping[str, object]) -> str:
    """Hash every retained field that can affect a migrated source snapshot."""
    fields = (
        'uuid',
        'kind',
        'url',
        'resolved_url',
        'title',
        'retrieved_at',
        'page_index',
        'line_start',
        'line_end',
        'raw_content',
        'content',
        'media_path',
        'media_type',
        'original_media_path',
        'original_media_type',
    )
    payload = {field: record[field] for field in fields}
    retrieved_at = payload['retrieved_at']
    if isinstance(retrieved_at, DateTime):
        payload['retrieved_at'] = retrieved_at.to_native().isoformat()
    elif isinstance(retrieved_at, datetime):
        payload['retrieved_at'] = retrieved_at.isoformat()
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()
