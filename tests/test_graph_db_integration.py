"""Opt-in Neo4j integration test. Gated on an EXPLICIT flag (``KMS_NEO4J_IT``), not on the mere
presence of ``NEO4J_URI`` — a configured ``.env`` (which ``db.py`` loads) would otherwise drag the
slow, network-dependent live tests into every ``pytest`` run. With the flag set it checks
connectivity, a round-trip query, and the structural-layer + entity-overlay writes against a real,
reachable instance whose creds come from ``NEO4J_URI``/``NEO4J_USERNAME``/``NEO4J_PASSWORD``.

Driven via asyncio.run so it needs no pytest-asyncio (the repo declares no such dev dep).
Run against a live DB with, e.g.:
    KMS_NEO4J_IT=1 NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \
        PYTHONPATH=src uv run pytest tests/test_graph_db_integration.py -q
"""

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("KMS_NEO4J_IT"),
    reason="set KMS_NEO4J_IT=1 (with NEO4J_URI/USERNAME/PASSWORD) to run the Neo4j integration test",
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
            meta = {"title": "Test Book", "author": "A. Mathematician"}
            await ensure_schema()
            await persist_nodes(stream, source, meta)
            await persist_nodes(stream, source, meta)  # idempotent re-run
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
                    "MATCH (s:Source {title: 'Test Book', author: 'A. Mathematician'})"
                    "-[:HEAD]->(n:Node) RETURN n.content AS c",
                )
                assert math["c"] == 1  # re-run did not duplicate the node
                assert chain["longest"] == 2
                assert head["c"] == "§1"  # title+author on the source, hangs off the first node
        finally:
            async with driver().session(database=database()) as session:
                await session.run("MATCH (n) DETACH DELETE n")  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())


def test_persist_entities_upserts_labels_root_and_members():
    from kms.core.models import ASTNode, Entity, EntityType, NodeType
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import persist_entities, persist_nodes

    source = "integration-test-book"
    stream = [
        ASTNode(type=NodeType.HEADER, content="§1", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="a right triangle is …", id=1, seg_index=0),
        ASTNode(type=NodeType.MATH, content="$a^2+b^2=c^2$", id=2, seg_index=0),
    ]
    overlay = [
        Entity(type=EntityType.DEFINITION, members=[1], id=0, title="Right triangle"),
        Entity(type=EntityType.THEOREM, members=[2], id=1, number="1.1"),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await ensure_schema()
            await persist_nodes(
                stream, source
            )  # the :Entity layer roots on the :Source/:Node layer
            await persist_entities(overlay, source)
            await persist_entities(overlay, source)  # idempotent re-run
            async with driver().session(database=database()) as session:
                # multi-label: the theorem is reachable as :Theorem and carries base :Entity too
                thm = await one(
                    session, "MATCH (e:Theorem:Entity {number: '1.1'}) RETURN count(e) AS c"
                )
                # both entities are rooted under the book's :Source via :HAS_ENTITY
                rooted = await one(
                    session,
                    "MATCH (:Source {key: '" + source + "'})-[:HAS_ENTITY]->(e:Entity) "
                    "RETURN count(e) AS c",
                )
                # the definition links to its member :Node (the paragraph) via :HAS_MEMBER
                member = await one(
                    session,
                    "MATCH (:Entity {title: 'Right triangle'})-[:HAS_MEMBER]->(n:Node) "
                    "RETURN n.content AS c",
                )
                assert thm["c"] == 1  # re-run did not duplicate the entity
                assert rooted["c"] == 2
                assert member["c"] == "a right triangle is …"
        finally:
            async with driver().session(database=database()) as session:
                await session.run("MATCH (n) DETACH DELETE n")  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())
