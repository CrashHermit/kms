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

One transport: the native ``neo4j`` async driver over Bolt
(``neo4j+s://…`` / ``bolt://…``, TCP port 7687) — the fast path and what a
self-hosted or Aura instance uses normally. The graph tier only ever does
``async with driver().session(database=...) as s: await s.run(cypher, **params)``
(plus ``verify_connectivity``/``close``), so a future transport swap means
changing just this module.

Async, to match the async pipeline (``asyncio.run(run(...))``, async stage
nodes). The driver holds a real pooled resource that must be closed, so we
keep one in an explicit module singleton with a ``close_driver`` teardown
(called from ``run()``'s ``finally`` once a stage opens a connection), instead
of an lru_cache that would have no place to close from.

The connection values map cleanly onto an Aura endpoint (``neo4j+s://…``) or a
self-hosted ``bolt://…`` instance, so swapping between them is just env vars.
"""

import os

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

# The shared native Bolt driver, created once and reused.
_driver: AsyncDriver | None = None


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


async def verify_connectivity() -> None:
    """Force a real handshake with the server.

    Surfaces auth/URI errors eagerly. Handy at startup and in the opt-in
    integration test.
    """
    await driver().verify_connectivity()


async def close_driver() -> None:
    """Close the shared driver and its connection pool, if one was opened.

    Idempotent; call from ``run()``'s ``finally`` so a run never leaks
    connections. Does not create a driver.
    """
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
