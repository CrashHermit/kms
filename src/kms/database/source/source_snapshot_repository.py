"""Persistence for deterministic, user-owned source snapshots."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession, AsyncTransaction

from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import SourceSnapshot
from kms.core.source_snapshot import assert_matching_snapshot
from kms.database.source.queries.source_pedagogical import LOAD_SOURCE_PAGES
from kms.database.source.queries.source_snapshot import (
    CREATE_OWNED_SNAPSHOT,
    LOCK_SNAPSHOT_OWNER,
    READ_OWNED_SNAPSHOT,
)
from kms.database.source.source_graph_records import (
    source_graph_parameters,
    source_pages_from_records,
)


class SourceSnapshotRepository:
    """Create and load ordinary source snapshots under a user's ownership."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_owned_snapshot(
        self, user_uuid: str, source_uuid: str
    ) -> SourceSnapshot | None:
        """Load one snapshot only through its owning user's relationship."""
        async with self._session_factory() as session:
            record = await self._read_owned_source(
                session, user_uuid, source_uuid
            )
            if record is None:
                return None
            page_result = await session.run(
                LOAD_SOURCE_PAGES, source_uuid=source_uuid
            )
            page_record = await page_result.single(strict=True)
        return self._snapshot(record, page_record['pages'])

    async def create_owned_snapshot(
        self,
        user_uuid: str,
        source: Source,
        pages: list[SourcePage],
    ) -> SourceSnapshot:
        """Atomically create an owned snapshot or return its concurrent winner."""
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._lock_user(transaction, user_uuid)
                existing = await self._read_owned_source(
                    transaction, user_uuid, source.uuid
                )
                first_write = existing is None
                if first_write:
                    await self._create(transaction, user_uuid, source, pages)
                    existing = await self._read_owned_source(
                        transaction, user_uuid, source.uuid
                    )
                page_result = await transaction.run(
                    LOAD_SOURCE_PAGES, source_uuid=source.uuid
                )
                page_record = await page_result.single(strict=True)
                snapshot = self._snapshot(existing, page_record['pages'])
                _assert_embedding_write(
                    source, pages, snapshot, first_write=first_write
                )
                assert_matching_snapshot(
                    SourceSnapshot(source=source, pages=pages), snapshot
                )
        return snapshot

    @staticmethod
    async def _lock_user(transaction: AsyncTransaction, user_uuid: str) -> None:
        result = await transaction.run(LOCK_SNAPSHOT_OWNER, user_uuid=user_uuid)
        await result.single(strict=True)

    @staticmethod
    async def _read_owned_source(
        executor: Any,
        user_uuid: str,
        source_uuid: str,
    ) -> dict[str, Any] | None:
        result = await executor.run(
            READ_OWNED_SNAPSHOT,
            user_uuid=user_uuid,
            source_uuid=source_uuid,
        )
        return await result.single(strict=False)

    @staticmethod
    async def _create(
        transaction: AsyncTransaction,
        user_uuid: str,
        source: Source,
        pages: list[SourcePage],
    ) -> None:
        parameters = source_graph_parameters(source, pages)
        result = await transaction.run(
            CREATE_OWNED_SNAPSHOT,
            user_uuid=user_uuid,
            **parameters,
        )
        await result.single(strict=True)

    @staticmethod
    def _snapshot(
        record: dict[str, Any], page_records: list[dict[str, Any]]
    ) -> SourceSnapshot:
        metadata = record['metadata']
        source = Source(
            uuid=record['uuid'],
            key=record['key'],
            metadata=json.loads(metadata)
            if isinstance(metadata, str)
            else metadata,
            revision=record['revision'],
        )
        return SourceSnapshot(
            source=source,
            pages=source_pages_from_records(page_records),
        )


def _assert_embedding_write(
    source: Source,
    pages: list[SourcePage],
    persisted: SourceSnapshot,
    *,
    first_write: bool,
) -> None:
    """Check profile reuse and the exact vectors stored on first write."""
    expected_profile = source.metadata.get('embedding_profile')
    persisted_profile = persisted.source.metadata.get('embedding_profile')
    if expected_profile != persisted_profile:
        message = (
            'Source embedding persistence mismatch'
            if first_write
            else 'Source embedding profile mismatch; reindex required'
        )
        raise ValueError(message)
    if not first_write:
        return

    submitted = [block.embedding for page in pages for block in page.blocks]
    stored = [
        block.embedding for page in persisted.pages for block in page.blocks
    ]
    if len(submitted) != len(stored):
        raise ValueError('Source embedding persistence mismatch')
    for submitted_vector, stored_vector in zip(submitted, stored, strict=True):
        if submitted_vector != stored_vector:
            raise ValueError('Source embedding persistence mismatch')
        if submitted_vector is not None and stored_vector is not None:
            if any(
                type(submitted_value) is not type(stored_value)
                for submitted_value, stored_value in zip(
                    submitted_vector, stored_vector, strict=True
                )
            ):
                raise ValueError('Source embedding persistence mismatch')
