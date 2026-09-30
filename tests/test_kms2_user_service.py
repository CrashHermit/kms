import asyncio

from fsrs import Scheduler

from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms2.database.user.user_repository import UserRepository
from kms2.service.user import UserService


class _Result:
    async def consume(self) -> None:
        return None

    async def data(self) -> list[dict[str, str]]:
        return []


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run(self, query: str, **parameters: object) -> _Result:
        self.calls.append((query, parameters))
        return _Result()


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_create_user_provisions_default_deck_with_fsrs_settings():
    session = _Session()

    def session_factory():
        return _SessionContext(session)

    service = UserService(
        UserRepository(session_factory),
        SourceCatalogRepository(session_factory),
    )

    user = asyncio.run(service.create_user('Alex'))

    assert user.name == 'Alex'
    _, parameters = session.calls[0]
    assert parameters['user_uuid'] == user.uuid
    assert parameters['user_name'] == 'Alex'
    assert parameters['deck_name'] == 'Default'
    scheduler_json = parameters['scheduler_json']
    assert isinstance(scheduler_json, str)
    assert Scheduler.from_json(scheduler_json).to_json() == scheduler_json
