"""Persistence access for users."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from neo4j import AsyncSession

from kms.users.models import User
from kms.users.queries import CREATE_USER, READ_USER, READ_USERS


class UserRepository:
    """Persist and list users."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_user(self, user: User) -> None:
        """Persist a user."""
        async with self._session_factory() as session:
            result = await session.run(
                CREATE_USER,
                user_uuid=user.uuid,
                user_name=user.name,
                created_at=user.created_at,
                updated_at=user.updated_at,
                invalidated_at=user.invalidated_at,
            )
            if result is not None:
                await result.consume()

    async def read_user(self, user_uuid: str) -> User:
        """Read exactly one user or propagate the strict database error."""
        async with self._session_factory() as session:
            result = await session.run(READ_USER, user_uuid=user_uuid)
            row = await result.single(strict=True)
        return User(
            uuid=row['uuid'],
            name=row['name'],
            created_at=_native_datetime(row['created_at']),
            updated_at=_native_datetime(row['updated_at']),
            invalidated_at=(
                _native_datetime(row['invalidated_at'])
                if row['invalidated_at'] is not None
                else None
            ),
        )

    async def list_users(self) -> list[User]:
        """Load users in stable display order."""
        async with self._session_factory() as session:
            result = await session.run(READ_USERS)
            rows = await result.data()
        return [
            User(
                uuid=row['uuid'],
                name=row['name'],
                created_at=_native_datetime(row['created_at']),
                updated_at=_native_datetime(row['updated_at']),
                invalidated_at=(
                    _native_datetime(row['invalidated_at'])
                    if row['invalidated_at'] is not None
                    else None
                ),
            )
            for row in rows
        ]


def _native_datetime(value: Any) -> datetime:
    """Convert Neo4j temporal values to Python datetimes."""
    return value if isinstance(value, datetime) else value.to_native()
