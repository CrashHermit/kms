"""
Neo4j connection for the graph tier — the ONLY module that imports a database client.

This is the graph tier's counterpart to ``core.llm``: credentials and client live in one
place, read from the environment, and every graph stage shares one instance. It lives in
``graph/`` rather than ``core/`` on purpose — ``core`` is the shared center every phase
depends on, and only phase 3 (and ``output`` exports, which may depend on ``graph``) touches
Neo4j. Keeping the client quarantined here also keeps the rest of the tier — models, matching
logic — pure and unit-testable without a database.

Two transports, one interface. The graph tier only ever does
``async with driver().session(database=...) as s: await s.run(cypher, **params)`` (plus
``verify_connectivity``/``close``), so both transports expose exactly that slice:

* **Bolt** (default) — the native ``neo4j`` async driver over ``neo4j+s://…`` / ``bolt://…``
  (TCP port 7687). This is the fast path and what a self-hosted or Aura instance uses normally.
* **HTTP Query API** — Aura's (and Neo4j 5's) Cypher-over-HTTPS endpoint
  (``POST https://<host>/db/<db>/query/v2``) on port 443. Bolt's 7687 is blocked in some
  sandboxed environments (only outbound HTTPS gets out); this transport reuses the *same*
  credentials over 443 so the tier still runs there. Selected with ``NEO4J_TRANSPORT=http``
  (or a ``http(s)://`` ``NEO4J_URI``); the HTTPS endpoint is derived from the same host, so no
  other config changes.

Async, to match the async pipeline (``asyncio.run(run(...))``, async stage nodes). Both
transports hold a real pooled/keep-alive resource that must be closed, so we keep one in an
explicit module singleton with a ``close_driver`` teardown (called from ``run()``'s ``finally``
once a stage opens a connection), instead of an lru_cache that would have no place to close from.

The four connection values map cleanly onto an Aura endpoint (``neo4j+s://…`` for Bolt or
``https://…`` for the Query API) and a self-hosted ``bolt://…`` instance, so swapping between
them is just these env vars.
"""

import os

import httpx
from neo4j import AsyncDriver, AsyncGraphDatabase

# Load a local .env if present, guarded — same convenience as core.llm, and harmless when a
# graph module is imported before any core.llm import has already loaded it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

URI_ENV = "NEO4J_URI"
USERNAME_ENV = "NEO4J_USERNAME"
PASSWORD_ENV = "NEO4J_PASSWORD"
DATABASE_ENV = "NEO4J_DATABASE"
TRANSPORT_ENV = "NEO4J_TRANSPORT"

# The shared client is either the native Bolt driver or the HTTP Query-API shim below; both
# expose the same ``session``/``verify_connectivity``/``close`` slice the tier consumes.
_driver: "AsyncDriver | HttpQueryDriver | None" = None


def _require(env_key: str, example: str) -> str:
    """Return the named connection value, raising a clear error if unset. Raised on use,
    not import, so graph modules stay importable without a database configured (the test
    suite and pure-logic paths never trip it)."""
    value = os.environ.get(env_key)
    if not value:
        raise RuntimeError(
            f"{env_key} is not set. Export it (e.g. `export {env_key}={example}`) "
            f"before running the graph tier."
        )
    return value


def is_configured() -> bool:
    """Whether a Neo4j target is configured (``NEO4J_URI`` set). Lets the pipeline skip graph
    persistence gracefully when no database is wired, so DB-less runs (and the test suite) still
    work end to end without a server."""
    return bool(os.environ.get(URI_ENV))


def database() -> str:
    """The target database name. Neo4j's default is ``neo4j``; Aura Free has exactly one."""
    return os.environ.get(DATABASE_ENV) or "neo4j"


# Internal alias so ``HttpQuerySession``/``HttpQueryDriver`` can resolve the default database
# without the ``database`` keyword parameter (the neo4j-compatible name) shadowing the function.
_default_database = database


def _use_http() -> bool:
    """Whether to talk to Neo4j over the HTTP Query API instead of Bolt. Explicit
    ``NEO4J_TRANSPORT`` wins (``http``/``https`` -> HTTP, ``bolt``/``neo4j`` -> Bolt); with it
    unset we infer from the URI scheme, so a plain ``https://…`` URI just works while the usual
    ``neo4j+s://…`` / ``bolt://…`` stays on Bolt."""
    transport = os.environ.get(TRANSPORT_ENV)
    if transport:
        return transport.strip().lower() in {"http", "https"}
    scheme = _require(URI_ENV, "neo4j+s://xxxx.databases.neo4j.io").split("://", 1)[0].lower()
    return scheme in {"http", "https"}


def _http_base_url() -> str:
    """Derive the HTTPS Query-API base (``https://host[:port]``) from ``NEO4J_URI``. Aura shares
    one host across Bolt and HTTP, so a ``neo4j+s://<host>`` URI yields ``https://<host>`` — the
    Query API is TLS on 443 — with no separate endpoint to configure. A ``http://…`` URI (a
    self-hosted HTTP endpoint) keeps its scheme and explicit port."""
    uri = _require(URI_ENV, "https://xxxx.databases.neo4j.io")
    scheme, _, rest = uri.partition("://")
    if not rest:  # no scheme in the URI — treat the whole thing as the host
        scheme, rest = "https", uri
    authority = rest.split("/", 1)[0].split("@")[-1]  # drop any path and inline credentials
    out_scheme = "http" if scheme.lower() == "http" else "https"
    return f"{out_scheme}://{authority}"


class HttpQueryError(RuntimeError):
    """A Cypher error surfaced by the HTTP Query API, carrying Neo4j's ``code`` so failures read
    like the Bolt driver's (auth, syntax, constraint, …) rather than a bare HTTP status."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class HttpQueryResult:
    """Minimal stand-in for a neo4j ``AsyncResult``: the returned rows as dicts, plus the
    ``single()`` coroutine the tier and its integration test consume. The structural writes
    discard the result entirely; only reads (``RETURN 1`` checks) touch these."""

    def __init__(self, records: list[dict]):
        self._records = records

    async def single(self) -> dict | None:
        return self._records[0] if self._records else None

    def __aiter__(self):
        async def gen():
            for record in self._records:
                yield record

        return gen()


class HttpQuerySession:
    """One transaction scope over the HTTP Query API. Mirrors a neo4j async session's slice:
    an ``async with`` block whose ``run`` executes a statement. Each ``run`` is an independent
    autocommit request — the same semantics as ``session.run`` over Bolt, which the writer
    already relies on (its MERGEs are ordered awaits, not one explicit transaction)."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, db: str):
        self._client = client
        self._url = f"{base_url}/db/{db}/query/v2"

    async def __aenter__(self) -> "HttpQuerySession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def run(self, query: str, **params) -> HttpQueryResult:
        resp = await self._client.post(self._url, json={"statement": query, "parameters": params})
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise
        errors = body.get("errors")
        if errors:
            first = errors[0]
            raise HttpQueryError(first.get("code", "Neo.Error"), first.get("message", resp.text))
        resp.raise_for_status()
        data = body.get("data") or {}
        fields = data.get("fields", [])
        values = data.get("values", [])
        return HttpQueryResult([dict(zip(fields, row, strict=False)) for row in values])


class HttpQueryDriver:
    """Bolt-shaped facade over the HTTP Query API, so the rest of the tier is transport-agnostic.
    Holds one keep-alive ``httpx.AsyncClient`` (the pool analogue) with Basic auth; ``trust_env``
    is left on so the client honours ``HTTPS_PROXY`` and ``SSL_CERT_FILE`` — the two knobs a
    sandboxed environment needs to reach Aura on 443."""

    def __init__(self, base_url: str, auth: tuple[str, str]):
        self._base_url = base_url
        self._client = httpx.AsyncClient(
            auth=auth,
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"Content-Type": "application/json"},
        )

    def session(self, database: str | None = None, **_) -> HttpQuerySession:
        return HttpQuerySession(self._client, self._base_url, database or _default_database())

    async def verify_connectivity(self) -> None:
        async with self.session() as s:
            await s.run("RETURN 1 AS ok")

    async def close(self) -> None:
        await self._client.aclose()


def driver() -> "AsyncDriver | HttpQueryDriver":
    """The shared client (Bolt driver or HTTP Query-API facade), created once and reused. Neither
    opens a socket at construction — Bolt connects lazily, the HTTP client on first request — so
    importing this module and calling ``driver()`` is safe without a live server; use
    ``verify_connectivity`` to force an actual handshake."""
    global _driver
    if _driver is None:
        uri = _require(URI_ENV, "neo4j+s://xxxx.databases.neo4j.io")  # required first
        auth = (
            _require(USERNAME_ENV, "neo4j"),
            _require(PASSWORD_ENV, "password"),
        )
        if _use_http():
            _driver = HttpQueryDriver(_http_base_url(), auth)
        else:
            _driver = AsyncGraphDatabase.driver(uri, auth=auth)
    return _driver


async def verify_connectivity() -> None:
    """Force a real handshake with the server, surfacing auth/URI errors eagerly. Handy at
    startup and in the opt-in integration test."""
    await driver().verify_connectivity()


async def close_driver() -> None:
    """Close the shared client and its connection pool, if one was opened. Idempotent; call
    from ``run()``'s ``finally`` so a run never leaks connections. Does not create a client."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
