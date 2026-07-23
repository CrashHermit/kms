"""Opt-in Neo4j integration test. Skipped unless NEO4J_URI is set (a real, reachable
instance), so it stays out of the hermetic unit suite — the same way live pipeline runs do.
With a database configured it checks connectivity and a trivial round-trip query.

Driven via asyncio.run so it needs no pytest-asyncio (the repo declares no such dev dep).
Run against a live DB with, e.g.:
    NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \
        PYTHONPATH=src uv run pytest tests/test_graph_db_integration.py -q
"""

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="set NEO4J_URI (and NEO4J_USERNAME/PASSWORD) to run the Neo4j integration test",
)


def test_connectivity_and_round_trip_query():
    from kms.graph.db import close_driver, database, driver, verify_connectivity

    async def scenario():
        try:
            await verify_connectivity()
            async with driver().session(database=database()) as session:
                result = await session.run("RETURN 1 AS n")
                record = await result.single()
                assert record["n"] == 1
        finally:
            await close_driver()

    asyncio.run(scenario())
