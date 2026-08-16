"""Neo4j access: driver, HTTP transport, sessions, and teardown.

Both the Bolt driver and a minimal HTTP/query-v2 transport are
supported. ``AUTO`` transport probes Bolt first and falls back to HTTP.
"""

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import ValidationError

from kms import config

BOLT = 'bolt'
HTTP = 'http'
AUTO = 'auto'
_driver: AsyncDriver | None = None
_http_client: httpx.AsyncClient | None = None
_probed_transport: str | None = None


def _db() -> config.DatabaseConfig:
    """Returns the database settings slice."""
    return config.get_settings().database


class Neo4jHTTPError(RuntimeError):
    """An error raised by the HTTP transport, with optional Neo4j code."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _require(value: str, name: str, example: str) -> str:
    """Returns a required configured value or raises.

    Args:
        value: The configured value.
        name: The environment variable the value came from.
        example: An example value for the error message.

    Returns:
        The configured value.

    Raises:
        RuntimeError: If the value is unset.
    """
    if not value:
        raise RuntimeError(
            f'{name} is not set. Export it '
            f'(e.g. `export {name}={example}`) '
            f'before running the graph tier.'
        )
    return value


def is_configured() -> bool:
    """True if a Neo4j URI is configured."""
    return bool(_db().uri)


def database() -> str:
    """Returns the configured database name, defaulting to 'neo4j'."""
    return _db().database


def configured_transport() -> str:
    """Returns the configured transport: bolt, http, or auto.

    Raises:
        RuntimeError: If the ``KMS_DATABASE__TRANSPORT`` value is invalid.
    """
    try:
        value = _db().transport.strip().lower()
    except ValidationError as error:
        if not any(
            'transport' in item.get('loc', ()) for item in error.errors()
        ):
            raise
        raise RuntimeError(
            'KMS_DATABASE__TRANSPORT is not valid. '
            f'Use {BOLT!r}, {HTTP!r}, or {AUTO!r}.'
        ) from error
    if value not in (BOLT, HTTP, AUTO):
        raise RuntimeError(
            f'KMS_DATABASE__TRANSPORT={value!r} is not valid. '
            f'Use {BOLT!r}, {HTTP!r}, or {AUTO!r}.'
        )
    return value


def http_url() -> str:
    """Returns the base HTTP URL for the configured Neo4j instance.

    Uses ``database.http_url`` when set, otherwise derives it from the
    Bolt URI (7474 for http, 7473 for https).
    """
    db = _db()
    explicit = db.http_url
    if explicit:
        return explicit.rstrip('/')

    uri = _require(
        db.uri, 'KMS_DATABASE__URI', 'neo4j+s://xxxx.databases.neo4j.io'
    )
    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        raise RuntimeError(
            f'KMS_DATABASE__URI={uri!r} has no host, so no HTTP URL can be '
            f'derived. Set KMS_DATABASE__HTTP_URL explicitly.'
        )

    scheme = parsed.scheme.lower()
    secure = '+s' in scheme
    if not secure:
        return f'http://{host}:{db.http_port}'
    return (
        f'https://{host}:{db.https_port}' if parsed.port else f'https://{host}'
    )


def query_endpoint() -> str:
    """Returns the HTTP query/v2 endpoint for the configured database."""
    return f'{http_url()}/db/{database()}/query/v2'


def driver() -> AsyncDriver:
    """Returns the shared Bolt driver, created lazily from the config."""
    global _driver
    if _driver is None:
        db = _db()
        uri = _require(
            db.uri, 'KMS_DATABASE__URI', 'neo4j+s://xxxx.databases.neo4j.io'
        )
        auth = (
            _require(db.username, 'KMS_DATABASE__USERNAME', 'neo4j'),
            _require(db.password, 'KMS_DATABASE__PASSWORD', 'password'),
        )
        _driver = AsyncGraphDatabase.driver(
            uri,
            auth=auth,
            max_connection_lifetime=db.max_connection_lifetime,
        )
    return _driver


def http_client() -> httpx.AsyncClient:
    """Returns the shared HTTP client with basic auth headers."""
    global _http_client
    if _http_client is None:
        db = _db()
        auth = httpx.BasicAuth(
            _require(db.username, 'KMS_DATABASE__USERNAME', 'neo4j'),
            _require(db.password, 'KMS_DATABASE__PASSWORD', 'password'),
        )
        _http_client = httpx.AsyncClient(
            auth=auth,
            timeout=httpx.Timeout(db.http_timeout),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
        )
    return _http_client


class HTTPResult:
    """A session-run result from the HTTP transport.

    Mirrors the small slice of the Bolt result API the graph code
    uses, so callers are transport-agnostic.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    async def all(self) -> list[dict[str, Any]]:
        """Returns all records as dicts."""
        return list(self._records)

    async def single(self, strict: bool = False) -> dict[str, Any] | None:
        """Returns the sole record, or None when there are none.

        Args:
            strict: When True, raise if the result is not exactly one
                record.

        Raises:
            Neo4jHTTPError: If strict and the record count differs.
        """
        if len(self._records) != 1 and strict:
            raise Neo4jHTTPError(
                f'expected exactly one record, got {len(self._records)}'
            )
        return self._records[0] if self._records else None

    async def data(self) -> list[dict[str, Any]]:
        """Returns copies of the records as plain dicts."""
        return [dict(record) for record in self._records]

    async def __aiter__(self):
        """Yields each record in order."""
        for record in self._records:
            yield record


class HTTPSession:
    """An async context manager that runs queries over the HTTP transport."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._bookmarks: list[str] = []

    async def __aenter__(self) -> 'HTTPSession':
        """Returns the session itself."""
        return self

    async def __aexit__(self, *exc_info) -> bool:
        """Closes nothing; the HTTP session is stateless."""
        return False

    async def run(self, query: str, **params: Any) -> HTTPResult:
        """Runs one query, returning decoded records.

        Args:
            query: The Cypher statement.
            **params: Named parameters for the statement.

        Returns:
            An HTTPResult over the returned records.

        Raises:
            Neo4jHTTPError: If the server reports errors or the HTTP
                status is an error status.
        """
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
        """Parses a JSON response body, tolerating non-JSON bodies."""
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}


class _LazySession:
    """Deferred session that resolves the auto transport on entry."""

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = kwargs
        self._inner: Any = None

    async def __aenter__(self):
        """Resolves the transport and enters the inner session."""
        self._inner = _session_for(await resolve_transport(), self._kwargs)
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc_info):
        """Delegates exit to the resolved inner session."""
        return await self._inner.__aexit__(*exc_info)


def _session_for(transport: str, kwargs: dict[str, Any]):
    """Builds a Bolt or HTTP session for the resolved transport."""
    if transport == BOLT:
        return driver().session(database=database(), **kwargs)
    return HTTPSession(http_client(), query_endpoint())


def session(**kwargs: Any):
    """Returns a session for the configured transport.

    With AUTO transport the actual session is deferred until the
    context is entered, so Bolt can be probed first.
    """
    transport = configured_transport()
    if transport == AUTO:
        return _LazySession(kwargs)
    return _session_for(transport, kwargs)


async def resolve_transport() -> str:
    """Resolves AUTO transport to bolt or http, caching the result.

    Bolt wins if its connectivity probe succeeds within the timeout;
    otherwise HTTP is used.
    """
    global _probed_transport

    configured = configured_transport()
    if configured != AUTO:
        return configured
    if _probed_transport is not None:
        return _probed_transport

    try:
        await asyncio.wait_for(
            driver().verify_connectivity(),
            timeout=_db().bolt_probe_timeout,
        )
    except Exception:
        await _close_bolt()
        _probed_transport = HTTP
    else:
        _probed_transport = BOLT
    return _probed_transport


async def verify_connectivity() -> None:
    """Verifies the resolved transport is reachable, raising on failure."""
    if await resolve_transport() == BOLT:
        await driver().verify_connectivity()
        return
    async with session() as live:
        await live.run('RETURN 1')


async def _close_bolt() -> None:
    """Closes and clears the shared Bolt driver."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def close_driver() -> None:
    """Closes the Bolt driver and HTTP client and resets transport cache."""
    global _http_client, _probed_transport
    await _close_bolt()
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _probed_transport = None
