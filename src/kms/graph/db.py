"""
Neo4j connection for the graph tier — the ONLY module that imports the driver.

This is the graph tier's counterpart to ``core.llm``: credentials and client live in one
place, read from the environment, and every graph stage shares one instance. It lives in
``graph/`` rather than ``core/`` on purpose — ``core`` is the shared center every phase
depends on, and only phase 3 (and ``output`` exports, which may depend on ``graph``) touches
Neo4j. Keeping the import quarantined here also keeps the rest of the tier — models, matching
logic — pure and unit-testable without a database.

Async, to match the async pipeline (``asyncio.run(run(...))``, async stage nodes): we use the
async driver so a stage can ``await`` reads/writes. The driver is a connection *pool*, so —
unlike ``core.llm``'s stateless ``@lru_cache`` LM config — it holds a real resource that must
be closed. We therefore keep it in an explicit module singleton with a ``close_driver``
teardown (called from ``run()``'s ``finally`` once a stage opens a connection), instead of an
lru_cache that would have no place to close from.

The four connection values map cleanly onto both an Aura ``neo4j+s://…`` endpoint and a
self-hosted ``bolt://…`` instance, so swapping between them is just these env vars.
"""

import os

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

_driver: AsyncDriver | None = None


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


def database() -> str:
    """The target database name. Neo4j's default is ``neo4j``; Aura Free has exactly one."""
    return os.environ.get(DATABASE_ENV) or "neo4j"


def driver() -> AsyncDriver:
    """The shared async driver (a connection pool), created once and reused. Construction
    does not open a socket — the driver connects lazily on first use — so importing this
    module and calling ``driver()`` is safe without a live server; use
    ``verify_connectivity`` to force an actual handshake."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            _require(URI_ENV, "neo4j+s://xxxx.databases.neo4j.io"),
            auth=(
                _require(USERNAME_ENV, "neo4j"),
                _require(PASSWORD_ENV, "password"),
            ),
        )
    return _driver


async def verify_connectivity() -> None:
    """Force a real handshake with the server, surfacing auth/URI errors eagerly. Handy at
    startup and in the opt-in integration test."""
    await driver().verify_connectivity()


async def close_driver() -> None:
    """Close the shared driver and its connection pool, if one was opened. Idempotent; call
    from ``run()``'s ``finally`` so a run never leaks connections. Does not create a driver."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
