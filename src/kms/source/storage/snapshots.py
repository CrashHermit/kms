"""Create and load complete v4 user-owned source snapshots."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession, AsyncTransaction

from kms.source.models.snapshot import (
    SOURCE_CONTENT_SCHEMA,
    SourceSnapshot,
    assert_matching_snapshot,
    require_source_schema,
)
from kms.source.models.source import Source
from kms.source.storage.records import (
    native_datetime,
    source_graph_parameters,
    source_units_from_records,
)
from kms.source.storage.snapshot_queries import (
    CREATE_OWNED_SNAPSHOT,
    LOCK_SNAPSHOT_OWNER,
    READ_OWNED_SNAPSHOT,
    READ_SOURCE_UNITS,
)


class SourceSnapshotRepository:
    """Create and load immutable source snapshots within ownership scope."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_owned_snapshot(
        self, user_uuid: str, source_uuid: str
    ) -> SourceSnapshot | None:
        async with self._session_factory() as session:
            record = await self._read_owned_source(
                session, user_uuid, source_uuid
            )
            if record is None:
                return None
            require_source_schema(record.get('identity_schema'))
            snapshot = await self._load_snapshot(session, record)
        return snapshot

    async def create_owned_snapshot(
        self, user_uuid: str, snapshot: SourceSnapshot
    ) -> SourceSnapshot:
        """Atomically publish a complete snapshot or return its winner."""
        require_source_schema(snapshot.source.metadata.get('identity_schema'))
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._lock_user(transaction, user_uuid)
                existing = await self._read_owned_source(
                    transaction, user_uuid, snapshot.source.uuid
                )
                first_write = existing is None
                if first_write:
                    await self._create(transaction, user_uuid, snapshot)
                    existing = await self._read_owned_source(
                        transaction, user_uuid, snapshot.source.uuid
                    )
                require_source_schema(existing.get('identity_schema'))
                persisted = await self._load_snapshot(transaction, existing)
                _assert_embedding_write(
                    snapshot, persisted, first_write=first_write
                )
                assert_matching_snapshot(snapshot, persisted)
        return persisted

    @staticmethod
    async def _lock_user(transaction: AsyncTransaction, user_uuid: str) -> None:
        result = await transaction.run(LOCK_SNAPSHOT_OWNER, user_uuid=user_uuid)
        await result.single(strict=True)

    @staticmethod
    async def _read_owned_source(
        executor: Any, user_uuid: str, source_uuid: str
    ):
        result = await executor.run(
            READ_OWNED_SNAPSHOT, user_uuid=user_uuid, source_uuid=source_uuid
        )
        return await result.single(strict=False)

    @staticmethod
    async def _load_snapshot(
        executor: Any, record: dict[str, Any]
    ) -> SourceSnapshot:
        result = await executor.run(
            READ_SOURCE_UNITS,
            source_uuid=record['uuid'],
            identity_schema=SOURCE_CONTENT_SCHEMA,
        )
        unit_record = await result.single(strict=True)
        metadata = record['metadata']
        source = Source(
            uuid=record['uuid'],
            key=record['key'],
            created_at=native_datetime(record['created_at']),
            updated_at=native_datetime(record['updated_at']),
            invalidated_at=native_datetime(record['invalidated_at'])
            if record['invalidated_at'] is not None
            else None,
            metadata=json.loads(metadata)
            if isinstance(metadata, str)
            else metadata,
            revision=record['revision'],
        )
        return SourceSnapshot(
            source=source,
            units=source_units_from_records(unit_record['units']),
        )

    @staticmethod
    async def _create(
        transaction: AsyncTransaction, user_uuid: str, snapshot: SourceSnapshot
    ) -> None:
        result = await transaction.run(
            CREATE_OWNED_SNAPSHOT,
            user_uuid=user_uuid,
            **source_graph_parameters(snapshot),
        )
        await result.single(strict=True)


def _assert_embedding_write(
    submitted: SourceSnapshot,
    persisted: SourceSnapshot,
    *,
    first_write: bool,
) -> None:
    """Check profile reuse and exact indexed vectors on first write."""
    expected_profile = submitted.source.metadata.get('embedding_profile')
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
    vectors = [
        [unit.embedding for unit in part.units]
        + [block.embedding for unit in part.units for block in unit.blocks]
        for part in (submitted, persisted)
    ]
    if vectors[0] != vectors[1]:
        raise ValueError('Source embedding persistence mismatch')
    for actual, stored in zip(*vectors, strict=True):
        if (
            actual is not None
            and stored is not None
            and any(
                type(left) is not type(right)
                for left, right in zip(actual, stored, strict=True)
            )
        ):
            raise ValueError('Source embedding persistence mismatch')
