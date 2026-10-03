"""Persistence access for users and their initial deck."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.user import Deck, User, UserSchedulerSettings
from kms2.database.user.queries.user import CREATE_USER, READ_USERS


class UserRepository:
    """Create users with their initial deck and scheduler settings."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_user(
        self,
        user: User,
        deck: Deck,
        settings: UserSchedulerSettings,
    ) -> None:
        """Persist the user, default deck, and settings atomically."""
        async with self._session_factory() as session:
            result = await session.run(
                CREATE_USER,
                user_uuid=user.uuid,
                user_name=user.name,
                deck_uuid=deck.uuid,
                deck_name=deck.name,
                settings_uuid=settings.uuid,
                scheduler_json=settings.scheduler_json,
            )
            if result is not None:
                await result.consume()

    async def list_users(self) -> list[User]:
        """Load users in stable display order."""
        async with self._session_factory() as session:
            result = await session.run(READ_USERS)
            rows = await result.data()
        return [User(uuid=row['uuid'], name=row['name']) for row in rows]
