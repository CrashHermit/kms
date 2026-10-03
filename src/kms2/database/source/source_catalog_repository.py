"""Persistence access for owned and unowned source catalogs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.source import Source
from kms2.database.source.queries.source_catalog import (
    ADOPT_SOURCE,
    READ_SOURCES,
    READ_UNOWNED_SOURCES,
)


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
        return [Source(uuid=row['uuid'], key=row['key']) for row in rows]

    async def list_unowned_sources(self) -> list[Source]:
        """Load sources that have no user owner."""
        async with self._session_factory() as session:
            result = await session.run(READ_UNOWNED_SOURCES)
            rows = await result.data()
        return [Source(uuid=row['uuid'], key=row['key']) for row in rows]

    async def adopt_source(self, user_uuid: str, source_uuid: str) -> None:
        """Claim an unowned source, rejecting a concurrent or prior claim."""
        async with self._session_factory() as session:
            result = await session.run(
                ADOPT_SOURCE,
                user_uuid=user_uuid,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        if not rows:
            raise SourceOwnershipConflict(source_uuid)
