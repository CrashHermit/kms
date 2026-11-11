import asyncio
from datetime import UTC, datetime

from kms.users.models import User
from kms.users.repository import UserRepository
from kms.users.service import UserService


class _Result:
    def __init__(self, row=None) -> None:
        self.row = row

    async def consume(self) -> None:
        return None

    async def single(self, *, strict: bool) -> dict[str, object]:
        assert strict
        return self.row

    async def data(self) -> list[dict[str, object]]:
        return [self.row] if self.row is not None else []


class _Neo4jDateTime:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def to_native(self) -> datetime:
        return self.value


class _Session:
    def __init__(self, result: _Result | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.result = result or _Result()

    async def run(self, query: str, **parameters: object) -> _Result:
        self.calls.append((query, parameters))
        return self.result


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_create_user_persists_only_the_user():
    session = _Session()

    def session_factory():
        return _SessionContext(session)

    service = UserService(UserRepository(session_factory))

    user = asyncio.run(service.create_user('Alex'))

    query, parameters = session.calls[0]
    assert 'CREATE (user:User' in query
    assert 'CREATE (deck:Deck' not in query
    assert 'UserSchedulerSettings' not in query
    assert parameters == {
        'user_uuid': user.uuid,
        'user_name': 'Alex',
        'created_at': user.created_at,
        'updated_at': user.created_at,
        'invalidated_at': None,
    }
    assert user.updated_at == user.created_at
    assert user.invalidated_at is None


def test_create_user_persists_explicit_lifecycle_timestamps():
    created_at = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    invalidated_at = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    user = User(
        uuid='user-uuid',
        name='Alex',
        created_at=created_at,
        updated_at=updated_at,
        invalidated_at=invalidated_at,
    )
    session = _Session()

    def session_factory():
        return _SessionContext(session)

    asyncio.run(UserRepository(session_factory).create_user(user))

    assert session.calls[0][1] == {
        'user_uuid': user.uuid,
        'user_name': user.name,
        'created_at': created_at,
        'updated_at': updated_at,
        'invalidated_at': invalidated_at,
    }


def test_read_user_preserves_lifecycle_timestamps():
    created_at = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    invalidated_at = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    session = _Session(
        _Result(
            {
                'uuid': 'user-uuid',
                'name': 'Alex',
                'created_at': _Neo4jDateTime(created_at),
                'updated_at': _Neo4jDateTime(updated_at),
                'invalidated_at': _Neo4jDateTime(invalidated_at),
            }
        )
    )

    def session_factory():
        return _SessionContext(session)

    user = asyncio.run(UserRepository(session_factory).read_user('user-uuid'))

    assert user == User(
        uuid='user-uuid',
        name='Alex',
        created_at=created_at,
        updated_at=updated_at,
        invalidated_at=invalidated_at,
    )


def test_list_users_preserves_lifecycle_timestamps_and_null_invalidation():
    created_at = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    session = _Session(
        _Result(
            {
                'uuid': 'user-uuid',
                'name': 'Alex',
                'created_at': _Neo4jDateTime(created_at),
                'updated_at': _Neo4jDateTime(updated_at),
                'invalidated_at': None,
            }
        )
    )

    def session_factory():
        return _SessionContext(session)

    users = asyncio.run(UserRepository(session_factory).list_users())

    assert users == [
        User(
            uuid='user-uuid',
            name='Alex',
            created_at=created_at,
            updated_at=updated_at,
        )
    ]
