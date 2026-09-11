"""KMS2-owned asynchronous Neo4j driver lifecycle."""

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from kms2.config import DatabaseSettings


class DatabaseClient:
    """Own one lazily created Neo4j driver for one database configuration."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._driver: AsyncDriver | None = None

    def driver(self) -> AsyncDriver:
        """Return the lazily created Neo4j driver for this client."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.uri,
                auth=(self._settings.username, self._settings.password),
            )
        return self._driver

    def session(self) -> AsyncSession:
        """Return an asynchronous session for the configured database."""
        return self.driver().session(database=self._settings.database)

    async def close(self) -> None:
        """Close this client's driver, when it has been created."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
