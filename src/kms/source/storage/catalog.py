"""Persistence access for owned and unowned source catalogs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms.core.models import Edge
from kms.source.models.snapshot import require_source_schema
from kms.source.models.source import Source
from kms.source.storage.catalog_queries import (
    ADOPT_SOURCE,
    READ_SOURCES,
    READ_UNOWNED_SOURCE_SCHEMA,
    READ_UNOWNED_SOURCES,
)
from kms.source.storage.records import native_datetime


class SourceOwnershipConflict(Exception):
    """Raised when a source cannot be claimed because it has an owner."""


class SourceCatalogRepository:
    """Access persisted source summaries and ownership relationships."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_sources(self, user_uuid: str) -> list[Source]:
        """Load one user's sources in stable display order."""
        async with self._session_factory() as session:
            result = await session.run(READ_SOURCES, user_uuid=user_uuid)
            rows = await result.data()
        return [
            Source(
                uuid=row['uuid'],
                key=row['key'],
                created_at=native_datetime(row['created_at']),
                updated_at=native_datetime(row['updated_at']),
                invalidated_at=native_datetime(row['invalidated_at'])
                if row['invalidated_at'] is not None
                else None,
                revision=row['revision'],
            )
            for row in rows
        ]

    async def list_unowned_sources(self) -> list[Source]:
        """Load sources that have no user owner."""
        async with self._session_factory() as session:
            result = await session.run(READ_UNOWNED_SOURCES)
            rows = await result.data()
        return [
            Source(
                uuid=row['uuid'],
                key=row['key'],
                created_at=native_datetime(row['created_at']),
                updated_at=native_datetime(row['updated_at']),
                invalidated_at=native_datetime(row['invalidated_at'])
                if row['invalidated_at'] is not None
                else None,
                revision=row['revision'],
            )
            for row in rows
        ]

    async def adopt_source(self, user_uuid: str, source_uuid: str) -> None:
        """Claim an unowned v3 source, rejecting concurrent or prior claims."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_UNOWNED_SOURCE_SCHEMA,
                source_uuid=source_uuid,
            )
            record = await result.single(strict=False)
            if record is None:
                raise SourceOwnershipConflict(source_uuid)
            require_source_schema(record.get('identity_schema'))
            result = await session.run(
                ADOPT_SOURCE,
                user_uuid=user_uuid,
                source_uuid=source_uuid,
                edge_created_at=Edge().created_at,
            )
            rows = await result.data()
        if not rows:
            raise SourceOwnershipConflict(source_uuid)
