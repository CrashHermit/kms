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


def test_connectivity_round_trip_and_idempotent_schema():
    from kms.graph.db import close_driver, database, driver, verify_connectivity
    from kms.graph.schema import ensure_schema

    async def scenario():
        try:
            await verify_connectivity()
            async with driver().session(database=database()) as session:
                result = await session.run("RETURN 1 AS n")
                record = await result.single()
                assert record["n"] == 1
            await ensure_schema()
            await ensure_schema()  # idempotent: a second pass must not raise
        finally:
            await close_driver()

    asyncio.run(scenario())


def test_persist_nodes_upserts_labels_and_next_chain():
    from kms.core.models import ASTNode, NodeType
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import persist_nodes

    source = "integration-test-book"
    stream = [
        ASTNode(type=NodeType.HEADER, content="§1", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="a", id=1, seg_index=0),
        ASTNode(type=NodeType.MATH, content="$x$", id=2, seg_index=0),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await ensure_schema()
            await persist_nodes(stream, source, {"title": "Test Book"})
            await persist_nodes(stream, source, {"title": "Test Book"})  # idempotent re-run
            async with driver().session(database=database()) as session:
                # multi-label: the math node is reachable as :Math and carries base :Node too
                math = await one(
                    session, "MATCH (n:Math:Node {content: '$x$'}) RETURN count(n) AS c"
                )
                # the :NEXT chain threads all three in order: §1 -> a -> $x$ (length 2)
                chain = await one(
                    session,
                    "MATCH p=(:Node)-[:NEXT*]->(:Node) RETURN max(length(p)) AS longest",
                )
                # the source roots the chain: :Source -HEAD-> the first node, and carries metadata
                head = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book'})-[:HEAD]->(n:Node) RETURN n.content AS c",
                )
                assert math["c"] == 1  # re-run did not duplicate the node
                assert chain["longest"] == 2
                assert head["c"] == "§1"  # hangs off the first structural node
        finally:
            async with driver().session(database=database()) as session:
                await session.run("MATCH (n) DETACH DELETE n")  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())
