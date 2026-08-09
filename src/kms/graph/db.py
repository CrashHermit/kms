import asyncio
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from neo4j import AsyncDriver, AsyncGraphDatabase

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

URI_ENV = 'NEO4J_URI'
USERNAME_ENV = 'NEO4J_USERNAME'
PASSWORD_ENV = 'NEO4J_PASSWORD'
DATABASE_ENV = 'NEO4J_DATABASE'
TRANSPORT_ENV = 'NEO4J_TRANSPORT'
HTTP_URL_ENV = 'NEO4J_HTTP_URL'
HTTP_TIMEOUT_ENV = 'NEO4J_HTTP_TIMEOUT'
BOLT_PROBE_TIMEOUT_ENV = 'NEO4J_BOLT_PROBE_TIMEOUT'

BOLT = 'bolt'
HTTP = 'http'
AUTO = 'auto'
HTTPS_PORT = 7473
HTTP_PORT = 7474
_driver: AsyncDriver | None = None
_http_client: httpx.AsyncClient | None = None
_probed_transport: str | None = None


class Neo4jHTTPError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _require(env_key: str, example: str) -> str:
    value = os.environ.get(env_key)
    if not value:
        raise RuntimeError(
            f'{env_key} is not set. Export it '
            f'(e.g. `export {env_key}={example}`) '
            f'before running the graph tier.'
        )
    return value


def _float_env(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def is_configured() -> bool:
    return bool(os.environ.get(URI_ENV))


def database() -> str:
    return os.environ.get(DATABASE_ENV) or 'neo4j'


def configured_transport() -> str:
    value = (os.environ.get(TRANSPORT_ENV) or AUTO).strip().lower()
    if value not in (BOLT, HTTP, AUTO):
        raise RuntimeError(
            f'{TRANSPORT_ENV}={value!r} is not valid. '
            f'Use {BOLT!r}, {HTTP!r}, or {AUTO!r}.'
        )
    return value


def http_url() -> str:
    explicit = os.environ.get(HTTP_URL_ENV)
    if explicit:
        return explicit.rstrip('/')

    uri = _require(URI_ENV, 'neo4j+s://xxxx.databases.neo4j.io')
    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        raise RuntimeError(
            f'{URI_ENV}={uri!r} has no host, so no HTTP URL can be derived. '
            f'Set {HTTP_URL_ENV} explicitly.'
        )

    scheme = parsed.scheme.lower()
    secure = '+s' in scheme
    if not secure:
        return f'http://{host}:{HTTP_PORT}'
    return f'https://{host}:{HTTPS_PORT}' if parsed.port else f'https://{host}'


def query_endpoint() -> str:
    return f'{http_url()}/db/{database()}/query/v2'


def driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        uri = _require(URI_ENV, 'neo4j+s://xxxx.databases.neo4j.io')
        auth = (
            _require(USERNAME_ENV, 'neo4j'),
            _require(PASSWORD_ENV, 'password'),
        )
        _driver = AsyncGraphDatabase.driver(uri, auth=auth)
    return _driver


def http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        auth = httpx.BasicAuth(
            _require(USERNAME_ENV, 'neo4j'),
            _require(PASSWORD_ENV, 'password'),
        )
        _http_client = httpx.AsyncClient(
            auth=auth,
            timeout=httpx.Timeout(_float_env(HTTP_TIMEOUT_ENV, 60.0)),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
        )
    return _http_client


class HTTPResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    async def all(self) -> list[dict[str, Any]]:
        return list(self._records)

    async def single(self, strict: bool = False) -> dict[str, Any] | None:
        if len(self._records) != 1 and strict:
            raise Neo4jHTTPError(
                f'expected exactly one record, got {len(self._records)}'
            )
        return self._records[0] if self._records else None

    async def data(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    async def __aiter__(self):
        for record in self._records:
            yield record


class HTTPSession:
    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._bookmarks: list[str] = []

    async def __aenter__(self) -> 'HTTPSession':
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def run(self, query: str, **params: Any) -> HTTPResult:
        payload: dict[str, Any] = {'statement': query}
        if params:
            payload['parameters'] = params
        if self._bookmarks:
            payload['bookmarks'] = self._bookmarks

        response = await self._client.post(self._endpoint, json=payload)
        body = self._decode(response)

        errors = body.get('errors') or []
        if errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            raise Neo4jHTTPError(
                f'{first.get("code", "unknown")}: '
                f'{first.get("message", errors)}',
                code=first.get('code'),
            )
        if response.status_code >= 400:
            raise Neo4jHTTPError(
                f'HTTP {response.status_code} from {self._endpoint}: '
                f'{response.text[:200]}'
            )

        bookmarks = body.get('bookmarks')
        if bookmarks:
            self._bookmarks = list(bookmarks)

        data = body.get('data') or {}
        fields = data.get('fields') or []
        values = data.get('values') or []
        return HTTPResult(
            [dict(zip(fields, row, strict=False)) for row in values]
        )

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}


class _LazySession:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = kwargs
        self._inner: Any = None

    async def __aenter__(self):
        self._inner = _session_for(await resolve_transport(), self._kwargs)
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc_info):
        return await self._inner.__aexit__(*exc_info)


def _session_for(transport: str, kwargs: dict[str, Any]):
    if transport == BOLT:
        return driver().session(database=database(), **kwargs)
    return HTTPSession(http_client(), query_endpoint())


def session(**kwargs: Any):
    transport = configured_transport()
    if transport == AUTO:
        return _LazySession(kwargs)
    return _session_for(transport, kwargs)


async def resolve_transport() -> str:
    global _probed_transport

    configured = configured_transport()
    if configured != AUTO:
        return configured
    if _probed_transport is not None:
        return _probed_transport

    try:
        await asyncio.wait_for(
            driver().verify_connectivity(),
            timeout=_float_env(BOLT_PROBE_TIMEOUT_ENV, 5.0),
        )
    except Exception:
        await _close_bolt()
        _probed_transport = HTTP
    else:
        _probed_transport = BOLT
    return _probed_transport


async def verify_connectivity() -> None:
    if await resolve_transport() == BOLT:
        await driver().verify_connectivity()
        return
    async with session() as live:
        await live.run('RETURN 1')


async def _close_bolt() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def close_driver() -> None:
    global _http_client, _probed_transport
    await _close_bolt()
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _probed_transport = None

