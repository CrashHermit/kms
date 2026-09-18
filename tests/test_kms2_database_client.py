import asyncio

from kms2.config.services import DatabaseSettings
from kms2.database.client import DatabaseClient


class _FakeDriver:
    def __init__(self) -> None:
        self.session_calls: list[dict[str, str]] = []
        self.close_calls = 0
        self.session_value = object()

    def session(self, **kwargs: str) -> object:
        self.session_calls.append(kwargs)
        return self.session_value

    async def close(self) -> None:
        self.close_calls += 1


def test_clients_keep_database_configuration_and_driver_state_separate(
    monkeypatch,
):
    created: list[tuple[str, tuple[str, str]]] = []
    drivers = {
        'bolt://first': _FakeDriver(),
        'bolt://second': _FakeDriver(),
    }

    def create_driver(uri: str, *, auth: tuple[str, str]) -> _FakeDriver:
        created.append((uri, auth))
        return drivers[uri]

    monkeypatch.setattr(
        'kms2.database.client.AsyncGraphDatabase.driver',
        create_driver,
    )
    first = DatabaseClient(
        DatabaseSettings(
            uri='bolt://first',
            username='first-user',
            password='first-password',
            database='first-db',
        )
    )
    second = DatabaseClient(
        DatabaseSettings(
            uri='bolt://second',
            username='second-user',
            password='second-password',
            database='second-db',
        )
    )

    async def exercise() -> None:
        assert first.driver() is drivers['bolt://first']
        assert first.driver() is drivers['bolt://first']
        assert second.driver() is drivers['bolt://second']
        assert first.session() is drivers['bolt://first'].session_value
        assert second.session() is drivers['bolt://second'].session_value
        await first.close()
        await second.close()

    asyncio.run(exercise())

    assert created == [
        ('bolt://first', ('first-user', 'first-password')),
        ('bolt://second', ('second-user', 'second-password')),
    ]
    assert drivers['bolt://first'].session_calls == [{'database': 'first-db'}]
    assert drivers['bolt://second'].session_calls == [{'database': 'second-db'}]
    assert drivers['bolt://first'].close_calls == 1
    assert drivers['bolt://second'].close_calls == 1
