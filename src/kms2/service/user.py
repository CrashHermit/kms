"""Application service for user and source ownership operations."""

from fsrs import Scheduler

from kms2.core.model.source import Source
from kms2.core.model.user import Deck, User, UserSchedulerSettings
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms2.database.user.user_repository import UserRepository


class UserService:
    """Coordinate user provisioning and source ownership repositories."""

    def __init__(
        self,
        users: UserRepository,
        sources: SourceCatalogRepository,
    ) -> None:
        self._users = users
        self._sources = sources

    async def create_user(self, name: str) -> User:
        """Create a user with a default deck and FSRS scheduler settings."""
        user = User(name=name)
        deck = Deck(name='Default')
        settings = UserSchedulerSettings(scheduler_json=Scheduler().to_json())
        await self._users.create_user(user, deck, settings)
        return user

    async def list_users(self) -> list[User]:
        """List users in stable display order."""
        return await self._users.list_users()

    async def list_sources(self, user_uuid: str) -> list[Source]:
        """List only sources owned by the requested user."""
        return await self._sources.list_sources(user_uuid)

    async def list_unowned_sources(self) -> list[Source]:
        """List legacy sources that have not been adopted."""
        return await self._sources.list_unowned_sources()

    async def adopt_source(self, user_uuid: str, source_uuid: str) -> None:
        """Explicitly transfer an unowned source to a user."""
        await self._sources.adopt_source(user_uuid, source_uuid)

    async def attach_new_source(
        self,
        user_uuid: str,
        source_uuid: str,
    ) -> None:
        """Associate a successfully processed new source with its user."""
        await self._sources.adopt_source(user_uuid, source_uuid)
