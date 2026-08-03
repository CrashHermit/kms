"""The Query API transport — pure, no server (httpx is driven through a
MockTransport, and the Bolt probe through a fake driver). Covers URL
derivation, the request/response shape of a statement, error surfacing,
bookmark chaining, and how `NEO4J_TRANSPORT` picks a transport."""

import asyncio
import json

import httpx
import pytest

from kms.graph import db

_CONN_ENV = {
    'NEO4J_URI': 'neo4j+s://xxxx.databases.neo4j.io',
    'NEO4J_USERNAME': 'neo4j',
    'NEO4J_PASSWORD': 'secret',
}


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Every test starts with no client, no driver, and no cached verdict."""
    for key, value in _CONN_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('NEO4J_TRANSPORT', raising=False)
    monkeypatch.delenv('NEO4J_HTTP_URL', raising=False)
    monkeypatch.delenv('NEO4J_DATABASE', raising=False)
    asyncio.run(db.close_driver())
    yield
    asyncio.run(db.close_driver())


def _client(handler) -> httpx.AsyncClient:
    """An httpx client answering from `handler` instead of the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _rows(fields, values, bookmarks=('bm-1',)):
    """A Query API success body."""
    return {
        'data': {'fields': list(fields), 'values': [list(v) for v in values]},
        'bookmarks': list(bookmarks),
    }


# -- URL derivation -------------------------------------------------------


def test_aura_uri_maps_to_plain_https(monkeypatch):
    monkeypatch.setenv('NEO4J_URI', 'neo4j+s://abc123.databases.neo4j.io')
    assert db.http_url() == 'https://abc123.databases.neo4j.io'


def test_self_hosted_bolt_uri_maps_to_the_http_port(monkeypatch):
    monkeypatch.setenv('NEO4J_URI', 'bolt://localhost:7687')
    assert db.http_url() == 'http://localhost:7474'


def test_secure_uri_with_an_explicit_port_maps_to_the_https_port(monkeypatch):
    # An explicit Bolt port never doubles as the HTTP port, so 7687 must not
    # survive into the derived URL.
    monkeypatch.setenv('NEO4J_URI', 'neo4j+ssc://graph.internal:7687')
    assert db.http_url() == 'https://graph.internal:7473'


def test_explicit_http_url_wins_and_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv('NEO4J_HTTP_URL', 'https://proxy.internal/neo4j/')
    assert db.http_url() == 'https://proxy.internal/neo4j'


def test_query_endpoint_targets_the_configured_database(monkeypatch):
    monkeypatch.setenv('NEO4J_DATABASE', 'kms')
    assert db.query_endpoint() == (
        'https://xxxx.databases.neo4j.io/db/kms/query/v2'
    )


def test_http_url_requires_a_uri(monkeypatch):
    monkeypatch.delenv('NEO4J_URI', raising=False)
    with pytest.raises(RuntimeError, match='NEO4J_URI is not set'):
        db.http_url()


# -- Statement round trip -------------------------------------------------


def test_run_posts_the_statement_and_maps_rows_to_dicts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['url'] = str(request.url)
        seen['body'] = json.loads(request.content)
        return httpx.Response(
            202, json=_rows(['uuid', 'name'], [['u1', 'a'], ['u2', 'b']])
        )

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                result = await s.run('MATCH (n) RETURN n', source='book')
                return await result.all()

    records = asyncio.run(scenario())
    assert seen['url'] == 'https://db/query/v2'
    assert seen['body'] == {
        'statement': 'MATCH (n) RETURN n',
        'parameters': {'source': 'book'},
    }
    # Records are plain mappings, so `record['k']` and `record.get('k')` — the
    # only two things queries.py asks of a Bolt record — both work.
    assert records == [
        {'uuid': 'u1', 'name': 'a'},
        {'uuid': 'u2', 'name': 'b'},
    ]
    assert records[0].get('missing') is None


def test_run_omits_parameters_when_there_are_none():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['body'] = json.loads(request.content)
        return httpx.Response(202, json=_rows([], []))

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                await s.run('CREATE CONSTRAINT x IF NOT EXISTS')

    asyncio.run(scenario())
    assert seen['body'] == {'statement': 'CREATE CONSTRAINT x IF NOT EXISTS'}


def test_single_returns_the_row_and_none_when_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if 'count' in body['statement']:
            return httpx.Response(202, json=_rows(['c'], [[3]]))
        return httpx.Response(202, json=_rows(['c'], []))

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                one = await (await s.run('RETURN count(n) AS c')).single()
                none = await (await s.run('RETURN nothing AS c')).single()
                return one, none

    one, none = asyncio.run(scenario())
    assert one == {'c': 3}
    assert none is None


def test_a_server_error_raises_with_the_neo4j_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                'errors': [
                    {
                        'code': 'Neo.ClientError.Database.DatabaseNotFound',
                        'message': "Database does not exist. Name: 'neo4j'.",
                    }
                ]
            },
        )

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                await s.run('RETURN 1')

    with pytest.raises(db.Neo4jHTTPError, match='DatabaseNotFound') as caught:
        asyncio.run(scenario())
    assert caught.value.code == 'Neo.ClientError.Database.DatabaseNotFound'


def test_a_non_json_failure_still_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text='<html>gateway</html>')

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                await s.run('RETURN 1')

    with pytest.raises(db.Neo4jHTTPError, match='HTTP 503'):
        asyncio.run(scenario())


def test_bookmarks_from_one_statement_are_replayed_on_the_next():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            202, json=_rows([], [], bookmarks=[f'bm-{len(bodies)}'])
        )

    async def scenario():
        async with _client(handler) as client:
            async with db.HTTPSession(client, 'https://db/query/v2') as s:
                await s.run('CREATE (:A)')
                await s.run('MATCH (a:A) RETURN a')

    asyncio.run(scenario())
    # Read-your-writes across statements in one session: the write's bookmark
    # rides along on the read that follows it.
    assert 'bookmarks' not in bodies[0]
    assert bodies[1]['bookmarks'] == ['bm-1']


# -- Transport selection --------------------------------------------------


class _FakeDriver:
    """A Bolt driver whose handshake we control."""

    def __init__(self, reachable: bool) -> None:
        self.reachable = reachable
        self.closed = False

    async def verify_connectivity(self):
        if not self.reachable:
            raise OSError('connection refused')

    async def close(self):
        self.closed = True

    def session(self, **kwargs):
        return ('bolt-session', kwargs)


def test_transport_defaults_to_auto_and_rejects_nonsense(monkeypatch):
    assert db.configured_transport() == 'auto'
    monkeypatch.setenv('NEO4J_TRANSPORT', 'HTTP')  # case-insensitive
    assert db.configured_transport() == 'http'
    monkeypatch.setenv('NEO4J_TRANSPORT', 'grpc')
    with pytest.raises(RuntimeError, match='not valid'):
        db.configured_transport()


def test_explicit_http_transport_yields_an_http_session(monkeypatch):
    monkeypatch.setenv('NEO4J_TRANSPORT', 'http')
    assert isinstance(db.session(), db.HTTPSession)


def test_explicit_bolt_transport_yields_a_bolt_session(monkeypatch):
    monkeypatch.setenv('NEO4J_TRANSPORT', 'bolt')
    monkeypatch.setenv('NEO4J_DATABASE', 'kms')
    monkeypatch.setattr(db, '_driver', _FakeDriver(reachable=True))
    assert db.session() == ('bolt-session', {'database': 'kms'})


def test_auto_uses_bolt_when_the_probe_succeeds(monkeypatch):
    monkeypatch.setattr(db, '_driver', _FakeDriver(reachable=True))
    assert asyncio.run(db.resolve_transport()) == 'bolt'


def test_auto_falls_back_to_http_when_bolt_is_unreachable(monkeypatch):
    driver = _FakeDriver(reachable=False)
    monkeypatch.setattr(db, '_driver', driver)

    async def scenario():
        transport = await db.resolve_transport()
        async with db.session() as live:
            return transport, live

    transport, live = asyncio.run(scenario())
    assert transport == 'http'
    assert isinstance(live, db.HTTPSession)
    # The unusable driver is closed rather than left holding a pool.
    assert driver.closed
    assert db._driver is None


def test_the_auto_probe_runs_once_per_process(monkeypatch):
    probes = []

    class _CountingDriver(_FakeDriver):
        async def verify_connectivity(self):
            probes.append(1)
            await super().verify_connectivity()

    monkeypatch.setattr(db, '_driver', _CountingDriver(reachable=True))
    asyncio.run(db.resolve_transport())
    asyncio.run(db.resolve_transport())
    assert len(probes) == 1


def test_close_driver_clears_the_cached_verdict(monkeypatch):
    monkeypatch.setattr(db, '_driver', _FakeDriver(reachable=False))
    assert asyncio.run(db.resolve_transport()) == 'http'
    asyncio.run(db.close_driver())
    assert db._probed_transport is None
    # A later run re-probes instead of trusting the stale verdict.
    monkeypatch.setattr(db, '_driver', _FakeDriver(reachable=True))
    assert asyncio.run(db.resolve_transport()) == 'bolt'


def test_close_driver_closes_the_http_client(monkeypatch):
    monkeypatch.setenv('NEO4J_TRANSPORT', 'http')
    client = db.http_client()
    assert db.http_client() is client  # reused, not rebuilt per session
    asyncio.run(db.close_driver())
    assert client.is_closed
    assert db._http_client is None
