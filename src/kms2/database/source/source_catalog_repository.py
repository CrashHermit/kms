"""Persistence access for the source catalog."""

from collections.abc import Callable

from kms2.core.model import Source
from kms2.database.source.queries.source_catalog import READ_SOURCES


class SourceCatalogRepository:
    """Access persisted source summaries."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def list_sources(self) -> list[Source]:
        """Load persisted sources in stable display order."""
        async with self._session_factory() as session:
            result = await session.run(READ_SOURCES)
            rows = await result.data()
        return [
            Source(
                uuid=row['uuid'],
                key=row['key'],
            )
            for row in rows
        ]
