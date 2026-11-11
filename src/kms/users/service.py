"""Application service for user identity operations."""

from kms.users.models import User
from kms.users.repository import UserRepository


class UserService:
    """Coordinate user identity persistence."""

    def __init__(self, users: UserRepository) -> None:
        self._users = users

    async def create_user(self, name: str) -> User:
        """Create and persist a user."""
        user = User(name=name)
        await self._users.create_user(user)
        return user

    async def list_users(self) -> list[User]:
        """List users in stable display order."""
        return await self._users.list_users()
