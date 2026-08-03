"""
Neo4j connection for the graph tier — the ONLY module that imports a database
client.

This is the graph tier's counterpart to ``core.llm``: credentials and client
live in one place, read from the environment, and every graph stage shares one
instance. It lives in ``graph/`` rather than ``core/`` on purpose — ``core`` is
the shared center every phase depends on, and only phase 3 (and ``output``
exports, which may depend on ``graph``) touches Neo4j. Keeping the client
quarantined here also keeps the rest of the tier — models, matching logic —
pure and unit-testable without a database.

Two transports, one call shape:

* **Bolt** — the native ``neo4j`` async driver (``neo4j+s://…`` / ``bolt://…``,
  TCP port 7687). The fast path, and what a self-hosted or Aura instance uses
  normally.
* **HTTP** — Neo4j's Query API v2 (``POST {base}/db/{database}/query/v2``) over
  ordinary HTTPS on 443. Slower and chattier, but it works from sandboxes that
  only allow egress on 443 (CI, Claude Code web sessions), where the Bolt port
  is unreachable.

The graph tier only ever does
``async with db.session() as s: await s.run(cypher, **params)`` (plus
``verify_connectivity``/``close_driver``), so both transports satisfy it and
callers never learn which one is live. ``NEO4J_TRANSPORT`` picks: ``bolt``,
``http``, or ``auto`` (the default — probe Bolt once, fall back to HTTP when
the port is blocked).

Async, to match the async pipeline (``asyncio.run(run(...))``, async stage
nodes). Both clients hold real pooled resources that must be closed, so we keep
them in explicit module singletons with a ``close_driver`` teardown (called
from ``run()``'s ``finally`` once a stage opens a connection), instead of an
lru_cache that would have no place to close from.

The connection values map cleanly onto an Aura endpoint (``neo4j+s://…``) or a
self-hosted ``bolt://…`` instance, so swapping between them is just env vars —
and the HTTP base URL is derived from the same ``NEO4J_URI``.
"""

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from neo4j import AsyncDriver, AsyncGraphDatabase

# Load a local .env if present, guarded — same convenience as core.llm, and
# harmless when a graph module is imported before any core.llm import has
# already loaded it.
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

# Neo4j's standard HTTP ports, used when NEO4J_URI names a Bolt port (7687)
# that says nothing about where the HTTP API listens.
HTTPS_PORT = 7473
HTTP_PORT = 7474

# The shared native Bolt driver, created once and reused.
_driver: AsyncDriver | None = None

# The shared HTTP client for the Query API, created once and reused.
_http_client: httpx.AsyncClient | None = None

# The transport `auto` settled on, cached so the Bolt probe runs once per
# process rather than once per session. Reset by close_driver.
_probed_transport: str | None = None


class Neo4jHTTPError(RuntimeError):
    """A Query API request failed.

    Carries the server's Neo4j error code when it sent one, so callers can
    tell a bad Cypher statement from a bad URL or bad credentials.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _require(env_key: str, example: str) -> str:
    """Return the named connection value, raising if it is unset.

    Raised on use, not import, so graph modules stay importable without a
    database configured (the test suite and pure-logic paths never trip it).

    Args:
        env_key: The environment variable to read.
        example: A sample value, quoted in the error message.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    value = os.environ.get(env_key)
    if not value:
        raise RuntimeError(
            f'{env_key} is not set. Export it '
            f'(e.g. `export {env_key}={example}`) '
            f'before running the graph tier.'
        )
    return value


def _float_env(env_key: str, default: float) -> float:
    """Read a positive float from the environment, falling back on nonsense."""
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def is_configured() -> bool:
    """Whether a Neo4j target is configured (``NEO4J_URI`` set).

    Lets the pipeline skip graph persistence gracefully when no database is
    wired, so DB-less runs (and the test suite) still work end to end without
    a server.

    Returns:
        True if a target URI is set.
    """
    return bool(os.environ.get(URI_ENV))


def database() -> str:
    """The target database name.

    Returns:
        ``NEO4J_DATABASE`` if set, else Neo4j's default ``neo4j`` (Aura Free
        has exactly one).
    """
    return os.environ.get(DATABASE_ENV) or 'neo4j'


def configured_transport() -> str:
    """The transport the environment asks for: ``bolt``, ``http``, or ``auto``.

    Returns:
        The normalised ``NEO4J_TRANSPORT`` value, defaulting to ``auto``.

    Raises:
        RuntimeError: If the variable is set to something else.
    """
    value = (os.environ.get(TRANSPORT_ENV) or AUTO).strip().lower()
    if value not in (BOLT, HTTP, AUTO):
        raise RuntimeError(
            f'{TRANSPORT_ENV}={value!r} is not valid. '
            f'Use {BOLT!r}, {HTTP!r}, or {AUTO!r}.'
        )
    return value


def http_url() -> str:
    """The Query API base URL, without a trailing slash.

    ``NEO4J_HTTP_URL`` wins when set — the escape hatch for a reverse proxy or
    a nonstandard port. Otherwise it is derived from ``NEO4J_URI``: the scheme
    decides TLS (``neo4j+s``/``bolt+s``/``+ssc`` → https), and the port is
    Neo4j's HTTP port rather than the Bolt port in the URI, since the two never
    coincide. A secure URI with no explicit port is the Aura shape, which
    serves the Query API on plain 443.

    Returns:
        A base URL such as ``https://xxxx.databases.neo4j.io``.

    Raises:
        RuntimeError: If neither variable yields a usable host.
    """
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
    secure = '+s' in scheme  # covers both `+s` and `+ssc`
    if not secure:
        return f'http://{host}:{HTTP_PORT}'
    # An explicit port means a self-hosted instance; Aura URIs carry none and
    # answer on 443.
    return f'https://{host}:{HTTPS_PORT}' if parsed.port else f'https://{host}'


def query_endpoint() -> str:
    """The full Query API v2 endpoint for the target database."""
    return f'{http_url()}/db/{database()}/query/v2'


def driver() -> AsyncDriver:
    """The shared Bolt driver.

    Created once and reused. The driver connects lazily, so importing this
    module and calling ``driver()`` is safe without a live server; use
    ``verify_connectivity`` to force an actual handshake.

    Returns:
        The shared driver.
    """
    global _driver
    if _driver is None:
        # Required first, so a missing URI is the error the caller sees.
        uri = _require(URI_ENV, 'neo4j+s://xxxx.databases.neo4j.io')
        auth = (
            _require(USERNAME_ENV, 'neo4j'),
            _require(PASSWORD_ENV, 'password'),
        )
        _driver = AsyncGraphDatabase.driver(uri, auth=auth)
    return _driver


def http_client() -> httpx.AsyncClient:
    """The shared HTTP client for the Query API.

    Created once and reused, so the Query API transport keeps a warm
    connection pool instead of paying a TLS handshake per statement.

    Returns:
        The shared client, carrying the configured basic auth.
    """
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
    """The rows of one Query API response.

    The Query API answers in full — there is no cursor to stream — so the rows
    are already in memory and the ``await``s exist only to match the Bolt
    result's shape.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    async def all(self) -> list[dict[str, Any]]:
        """Every row, each a ``{column: value}`` mapping."""
        return list(self._records)

    async def single(self, strict: bool = False) -> dict[str, Any] | None:
        """The one row, mirroring the Bolt result's non-strict default.

        Args:
            strict: Raise when the result does not hold exactly one row,
                instead of returning the first (or None).

        Returns:
            The row, or None when the result is empty and ``strict`` is False.

        Raises:
            Neo4jHTTPError: If ``strict`` and the row count is not 1.
        """
        if len(self._records) != 1 and strict:
            raise Neo4jHTTPError(
                f'expected exactly one record, got {len(self._records)}'
            )
        return self._records[0] if self._records else None

    async def data(self) -> list[dict[str, Any]]:
        """Every row as a plain dict — the Bolt result's ``data()``."""
        return [dict(record) for record in self._records]

    async def __aiter__(self):
        for record in self._records:
            yield record


class HTTPSession:
    """A session-shaped wrapper over the Query API.

    Each ``run`` is one POST, which the server executes as its own implicit
    transaction — the same auto-commit semantics a Bolt ``session.run`` has.
    Bookmarks returned by one request are replayed on the next, so a session's
    statements keep read-your-writes ordering across the cluster exactly as a
    Bolt session does; that is what makes ``ensure_schema`` followed
    immediately by ``persist_nodes`` safe here.
    """

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
        # Nothing to release: the pooled client outlives the session, and each
        # statement already committed on its own.
        return False

    async def run(self, query: str, **params: Any) -> HTTPResult:
        """Execute one Cypher statement.

        Args:
            query: The Cypher statement.
            **params: Query parameters, JSON-encoded as-is. The graph tier
                passes only JSON-native values (strings, numbers, bools, and
                lists/dicts of them), which is what the Query API accepts.

        Returns:
            The rows the statement produced.

        Raises:
            Neo4jHTTPError: If the server reported an error or a non-2xx
                status.
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
        """The response body as a mapping, tolerating non-JSON error pages."""
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}


class _LazySession:
    """A session that picks its transport when it is entered.

    ``auto`` cannot decide synchronously — probing Bolt is an ``await`` — but
    the graph tier's factories are plain callables (``session_factory()``
    inside an ``async with``). Deferring the choice to ``__aenter__`` keeps
    that contract intact.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = kwargs
        self._inner: Any = None

    async def __aenter__(self):
        self._inner = _session_for(await resolve_transport(), self._kwargs)
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc_info):
        return await self._inner.__aexit__(*exc_info)


def _session_for(transport: str, kwargs: dict[str, Any]):
    """Build a session on an already-resolved transport."""
    if transport == BOLT:
        return driver().session(database=database(), **kwargs)
    return HTTPSession(http_client(), query_endpoint())


def session(**kwargs: Any):
    """A session on whichever transport is configured.

    The one entry point the graph tier should use: it yields an async context
    manager whose ``run(cypher, **params)`` behaves the same over Bolt and over
    the Query API.

    Args:
        **kwargs: Extra Bolt session arguments (``default_access_mode`` and
            friends). Ignored by the HTTP transport, which has no session
            object on the server to configure.

    Returns:
        An async context manager yielding the session.
    """
    transport = configured_transport()
    if transport == AUTO:
        return _LazySession(kwargs)
    return _session_for(transport, kwargs)


async def resolve_transport() -> str:
    """The transport to actually use, probing Bolt once under ``auto``.

    An explicit ``NEO4J_TRANSPORT`` is taken at its word. Under ``auto`` this
    opens a real Bolt handshake, bounded by ``NEO4J_BOLT_PROBE_TIMEOUT``
    (default 5s): if it succeeds Bolt wins, and if the port is blocked or the
    handshake fails the Query API takes over. The verdict is cached until
    ``close_driver``, so the probe costs one round trip per process, not one
    per session.

    Returns:
        Either ``bolt`` or ``http``.
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
            timeout=_float_env(BOLT_PROBE_TIMEOUT_ENV, 5.0),
        )
    except Exception:
        # Bolt is unreachable (blocked port, wrong scheme, dead host). Drop the
        # half-built driver so its pool is not left dangling, then fall back —
        # a real problem with the credentials or URL will resurface on the
        # first HTTP statement, with the server's own error message.
        await _close_bolt()
        _probed_transport = HTTP
    else:
        _probed_transport = BOLT
    return _probed_transport


async def verify_connectivity() -> None:
    """Force a real handshake with the server, on the live transport.

    Surfaces auth/URI errors eagerly. Handy at startup and in the opt-in
    integration test.
    """
    if await resolve_transport() == BOLT:
        await driver().verify_connectivity()
        return
    async with session() as live:
        await live.run('RETURN 1')


async def _close_bolt() -> None:
    """Close the Bolt driver if one was opened. Idempotent."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def close_driver() -> None:
    """Close both clients and their pools, if either was opened.

    Idempotent; call from ``run()``'s ``finally`` so a run never leaks
    connections. Creates nothing, and clears the cached ``auto`` verdict so a
    later run re-probes rather than trusting a stale one.
    """
    global _http_client, _probed_transport
    await _close_bolt()
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _probed_transport = None
