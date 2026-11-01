"""Persistence for deterministic, user-owned source snapshots."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession, AsyncTransaction

from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import OwnedSourceSnapshot
from kms2.database.source.queries.source_pedagogical import LOAD_SOURCE_PAGES
from kms2.database.source.queries.source_snapshot import (
    CREATE_OWNED_SNAPSHOT,
    LOCK_SNAPSHOT_OWNER,
    READ_OWNED_SNAPSHOT,
)
from kms2.database.source.source_graph_records import (
    source_graph_parameters,
    source_pages_from_records,
)


class SourceSnapshotRepository:
    """Create and load ordinary source snapshots under a user's ownership."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_owned_snapshot(
        self, user_uuid: str, source_uuid: str
    ) -> OwnedSourceSnapshot | None:
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
    ) -> OwnedSourceSnapshot:
        """Atomically create an owned snapshot or return its concurrent winner."""
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._lock_user(transaction, user_uuid)
                existing = await self._read_owned_source(
                    transaction, user_uuid, source.uuid
                )
                if existing is None:
                    await self._create(transaction, user_uuid, source, pages)
                    existing = await self._read_owned_source(
                        transaction, user_uuid, source.uuid
                    )
                page_result = await transaction.run(
                    LOAD_SOURCE_PAGES, source_uuid=source.uuid
                )
                page_record = await page_result.single(strict=True)
        return self._snapshot(existing, page_record['pages'])

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
    ) -> OwnedSourceSnapshot:
        metadata = record['metadata']
        source = Source(
            uuid=record['uuid'],
            key=record['key'],
            metadata=json.loads(metadata)
            if isinstance(metadata, str)
            else metadata,
            revision=record['revision'],
        )
        return OwnedSourceSnapshot(
            source=source,
            pages=source_pages_from_records(page_records),
        )


async def create_owned_snapshot_in_transaction(
    transaction: AsyncTransaction,
    user_uuid: str,
    source: Source,
    pages: list[SourcePage],
) -> OwnedSourceSnapshot:
    """Create or return an owned snapshot inside the caller's transaction."""
    await SourceSnapshotRepository._lock_user(transaction, user_uuid)
    existing = await SourceSnapshotRepository._read_owned_source(
        transaction, user_uuid, source.uuid
    )
    if existing is None:
        await SourceSnapshotRepository._create(
            transaction, user_uuid, source, pages
        )
        existing = await SourceSnapshotRepository._read_owned_source(
            transaction, user_uuid, source.uuid
        )
    page_result = await transaction.run(
        LOAD_SOURCE_PAGES, source_uuid=source.uuid
    )
    page_record = await page_result.single(strict=True)
    return SourceSnapshotRepository._snapshot(existing, page_record['pages'])


async def read_owned_snapshot_in_transaction(
    transaction: AsyncTransaction,
    user_uuid: str,
    source_uuid: str,
) -> OwnedSourceSnapshot:
    """Reload one caller-owned source and its pages inside the transaction."""
    result = await transaction.run(
        READ_OWNED_SNAPSHOT,
        user_uuid=user_uuid,
        source_uuid=source_uuid,
    )
    record = await result.single(strict=True)
    page_result = await transaction.run(
        LOAD_SOURCE_PAGES,
        source_uuid=source_uuid,
    )
    page_record = await page_result.single(strict=True)
    return SourceSnapshotRepository._snapshot(record, page_record['pages'])
