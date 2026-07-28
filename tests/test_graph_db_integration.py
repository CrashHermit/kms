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
    not os.environ.get('KMS_NEO4J_IT'),
    reason='set KMS_NEO4J_IT=1 (with NEO4J_URI/USERNAME/PASSWORD) to run the Neo4j integration test',
)


def test_connectivity_round_trip_and_idempotent_schema():
    from kms.graph import db, schema

    async def scenario():
        try:
            await db.verify_connectivity()
            async with db.driver().session(database=db.database()) as session:
                result = await session.run('RETURN 1 AS n')
                record = await result.single()
                assert record['n'] == 1
            await schema.ensure_schema()
            await schema.ensure_schema()  # idempotent: a second pass must not raise
        finally:
            await db.close_driver()

    asyncio.run(scenario())


def test_persist_nodes_upserts_labels_and_next_chain():
    from kms.core import models
    from kms.graph import db, schema, writer

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.HeaderNode, content='§1', id=0, segment_index=0
        ),
        models.ASTNode(
            type=models.ParagraphNode, content='a', id=1, segment_index=0
        ),
        models.ASTNode(
            type=models.MathNode, content='$x$', id=2, segment_index=0
        ),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            meta = {'title': 'Test Book', 'author': 'A. Mathematician'}
            await schema.ensure_schema()
            await writer.persist_nodes(stream, source, meta)
            await writer.persist_nodes(stream, source, meta)  # idempotent re-run
            async with db.driver().session(database=db.database()) as session:
                # multi-label: the math node is reachable as :Math and carries base :Node too
                math = await one(
                    session,
                    "MATCH (n:Math:Node {content: '$x$'}) RETURN count(n) AS c",
                )
                # the :NEXT chain threads all three in order: §1 -> a -> $x$ (length 2)
                chain = await one(
                    session,
                    'MATCH p=(:Node)-[:NEXT*]->(:Node) RETURN max(length(p)) AS longest',
                )
                # the source roots the chain: :Source -HEAD-> the first node, and carries metadata
                head = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book', author: 'A. Mathematician'})"
                    '-[:HEAD]->(n:Node) RETURN n.content AS c',
                )
                assert math['c'] == 1  # re-run did not duplicate the node
                assert chain['longest'] == 2
                assert (
                    head['c'] == '§1'
                )  # title+author on the source, hangs off the first node
        finally:
            async with db.driver().session(database=db.database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await db.close_driver()

    asyncio.run(scenario())


def test_persist_entities_and_procedures_upsert_the_overlay_and_its_spine():
    from kms.core import models
    from kms.graph import db, nodes, schema, writer

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.HeaderNode, content='§1', id=0, segment_index=0
        ),
        models.ASTNode(
            type=models.ParagraphNode,
            content='a right triangle is …',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.ParagraphNode,
            content='Theorem 1.1. $a^2+b^2=c^2$',
            id=2,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.ParagraphNode,
            content='Proof. Drop the altitude. Then compare areas.',
            id=3,
            segment_index=0,
        ),
    ]
    # A definition, and a theorem whose proof is its own span with two steps. Decomposition is
    # universal and the procedure carries its own :DERIVED_FROM provenance.
    overlay = [
        models.Entity(
            type='definition', members=[1], id=0, title='Right Triangle'
        ),
        models.Entity(
            type='theorem',
            members=[2],
            id=1,
            number='1.1',
            title='Pythagorean Theorem',
            procedures=[
                models.Procedure(
                    index=0,
                    members=[3],
                    contents=['Drop the altitude. Then compare areas.'],
                    steps=['Drop the altitude.', 'Then compare areas.'],
                )
            ],
        ),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await schema.ensure_schema()
            await writer.persist_nodes(stream, source)
            await writer.persist_entities(overlay, source)
            await writer.persist_procedures(overlay, source)
            await writer.persist_entities(overlay, source)  # idempotent re-run
            await writer.persist_procedures(overlay, source)
            async with db.driver().session(database=db.database()) as session:
                # type is an open PROPERTY on a bare :Entity — no per-type label is minted
                typed = await one(
                    session,
                    "MATCH (e:Entity {type: 'theorem'}) RETURN count(e) AS c",
                )
                labelled = await one(
                    session, 'MATCH (e:Theorem) RETURN count(e) AS c'
                )
                # the overlay is linked to its book via the source property and points back
                # at its member chunks
                src_uuid = nodes.source_uuid(source)
                rooted = await one(
                    session,
                    f"MATCH (e:Entity {{source: '{src_uuid}'}}) RETURN count(e) AS c",
                )
                derived = await one(
                    session,
                    'MATCH (:Entity)-[:DERIVED_FROM]->(:Node) RETURN count(*) AS c',
                )
                # the procedural spine: one :Procedure, its own provenance, and a 2-act chain
                spine = await one(
                    session,
                    'MATCH (:Entity)-[:HAS_PROCEDURE]->(p:Procedure)-[:FIRST]->(a:Act)'
                    '-[:THEN]->(b:Act) RETURN a.text AS first, b.text AS second',
                )
                proc_prov = await one(
                    session,
                    'MATCH (:Procedure)-[:DERIVED_FROM]->(:Node) RETURN count(*) AS c',
                )
                assert typed['c'] == 1  # re-run did not duplicate
                assert labelled['c'] == 0  # open types never become labels
                assert rooted['c'] == 2
                assert derived['c'] == 2
                assert spine['first'] == 'Drop the altitude.'
                assert spine['second'] == 'Then compare areas.'
                assert proc_prov['c'] == 1
        finally:
            async with db.driver().session(database=db.database()) as session:
                await session.run('MATCH (n) DETACH DELETE n')
            await db.close_driver()

    asyncio.run(scenario())
