import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('KMS_NEO4J_IT'),
    reason='set KMS_NEO4J_IT=1 (with NEO4J_URI/USERNAME/PASSWORD) '
    'to run the Neo4j integration test',
)


def test_connectivity_round_trip_and_idempotent_schema():
    from kms.graph import db, schema

    async def scenario():
        def _session_factory():
            return db.session()

        try:
            await db.verify_connectivity()
            async with db.session() as session:
                result = await session.run('RETURN 1 AS n')
                record = await result.single()
                assert record['n'] == 1
            await schema.ensure_schema(_session_factory)
            await schema.ensure_schema(
                _session_factory
            )
        finally:
            await db.close_driver()

    asyncio.run(scenario())


def test_persist_nodes_upserts_labels_and_next_chain():
    from kms.core import models
    from kms.graph import db, schema, writer

    source = 'integration-test-book'
    stream = [
        models.ASTNode(type='header', content='§1', id=0, segment_index=0),
        models.ASTNode(type='paragraph', content='a', id=1, segment_index=0),
        models.ASTNode(type='math', content='$x$', id=2, segment_index=0),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        def _session_factory():
            return db.session()

        try:
            meta = {'title': 'Test Book', 'author': 'A. Mathematician'}
            await schema.ensure_schema(_session_factory)
            await writer.persist_nodes(
                stream,
                source,
                session_factory=_session_factory,
                metadata=meta,
            )
            await writer.persist_nodes(
                stream,
                source,
                session_factory=_session_factory,
                metadata=meta,
            )
            await writer.persist_chain(
                stream, source, session_factory=_session_factory
            )
            await writer.persist_chain(
                stream, source, session_factory=_session_factory
            )
            async with db.session() as session:
                math = await one(
                    session,
                    "MATCH (n:Math:Node {content: '$x$'}) RETURN count(n) AS c",
                )
                chain = await one(
                    session,
                    'MATCH p=(:Node)-[:NEXT*]->(:Node) '
                    'RETURN max(length(p)) AS longest',
                )
                head = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book', "
                    "author: 'A. Mathematician'})"
                    '-[:HEAD]->(n:Node) RETURN n.content AS c',
                )
                assert math['c'] == 1
                assert chain['longest'] == 2
                assert (
                    head['c'] == '§1'
                )
        finally:
            async with db.session() as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )
            await db.close_driver()

    asyncio.run(scenario())

