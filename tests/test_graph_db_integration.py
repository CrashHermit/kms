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
    from kms.graph.db import close_driver, database, driver, verify_connectivity
    from kms.graph.schema import ensure_schema

    async def scenario():
        try:
            await verify_connectivity()
            async with driver().session(database=database()) as session:
                result = await session.run('RETURN 1 AS n')
                record = await result.single()
                assert record['n'] == 1
            await ensure_schema()
            await ensure_schema()  # idempotent: a second pass must not raise
        finally:
            await close_driver()

    asyncio.run(scenario())


def test_persist_nodes_upserts_labels_and_next_chain():
    from kms.core import models
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import persist_nodes

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.NodeType.HEADER, content='§1', id=0, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content='a', id=1, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.MATH, content='$x$', id=2, segment_index=0
        ),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            meta = {'title': 'Test Book', 'author': 'A. Mathematician'}
            await ensure_schema()
            await persist_nodes(stream, source, meta)
            await persist_nodes(stream, source, meta)  # idempotent re-run
            async with driver().session(database=database()) as session:
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
            async with driver().session(database=database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())


def test_persist_entities_upserts_labels_root_and_members():
    from kms.core import models
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import persist_entities, persist_nodes

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.NodeType.HEADER, content='§1', id=0, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='a right triangle is …',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.MATH,
            content='$a^2+b^2=c^2$',
            id=2,
            segment_index=0,
        ),
    ]
    overlay = [
        models.Entity(
            type='definition',
            members=[1],
            id=0,
            title='Right triangle',
        ),
        models.Entity(type='theorem', members=[2], id=1, number='1.1'),
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
                # the type is a property on the base :Entity label, not a per-type label
                thm = await one(
                    session,
                    "MATCH (e:Entity {type: 'theorem', number: '1.1'}) "
                    'RETURN count(e) AS c',
                )
                # both entities are rooted under the book's :Source via :HAS_ENTITY
                rooted = await one(
                    session,
                    "MATCH (:Source {key: '"
                    + source
                    + "'})-[:HAS_ENTITY]->(e:Entity) "
                    'RETURN count(e) AS c',
                )
                # the definition links to its member :Node (the paragraph) via :DERIVED_FROM
                member = await one(
                    session,
                    "MATCH (:Entity {title: 'Right triangle'})-[:DERIVED_FROM]->(n:Node) "
                    'RETURN n.content AS c',
                )
                assert thm['c'] == 1  # re-run did not duplicate the entity
                assert rooted['c'] == 2
                assert member['c'] == 'a right triangle is …'
        finally:
            async with driver().session(database=database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())


def test_persist_references_mints_hubs_and_edges():
    from kms.core import models
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import (
        persist_entities,
        persist_nodes,
        persist_references,
    )

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content='…', id=0, segment_index=0
        )
    ]
    # Two entities that both reference "Set" — they must converge on ONE hub.
    overlay = [
        models.Entity(
            type='theorem',
            members=[0],
            id=0,
            refs=[
                models.Reference(
                    target='Set', kind='definition', relation='assumes'
                )
            ],
        ),
        models.Entity(
            type='problem',
            members=[0],
            id=1,
            refs=[
                models.Reference(
                    target='set', kind='definition', relation='applies'
                )
            ],
        ),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await ensure_schema()
            await persist_nodes(stream, source)
            await persist_entities(overlay, source)
            await persist_references(overlay, source)
            await persist_references(overlay, source)  # idempotent re-run
            async with driver().session(database=database()) as session:
                # "Set" and "set" collapse to a single canonical
                hubs = await one(
                    session,
                    "MATCH (c:Canonical {type: 'definition'}) RETURN count(c) AS c",
                )
                # both entities reference that one canonical -> two :REFERENCES edges into it
                edges = await one(
                    session,
                    "MATCH (:Entity)-[r:REFERENCES]->(:Canonical {name: 'Set'}) "
                    'RETURN count(r) AS c',
                )
                # the open relation rides on the relationship
                relation = await one(
                    session,
                    "MATCH (:Entity {type: 'theorem'})-[r:REFERENCES]->(:Canonical) "
                    'RETURN r.relation AS t',
                )
                assert hubs['c'] == 1  # converged, and re-run did not duplicate
                assert edges['c'] == 2
                assert relation['t'] == 'assumes'
        finally:
            async with driver().session(database=database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())


def test_persist_realizes_wires_a_mention_to_the_canonical_it_defines():
    from kms.core import models
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import (
        persist_entities,
        persist_nodes,
        persist_realizes,
        persist_references,
    )

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content='…', id=0, segment_index=0
        )
    ]
    # A definition of "Vector Space", a theorem that references it, and a definition nobody cites.
    overlay = [
        models.Entity(
            type='definition',
            members=[0],
            id=0,
            title='Vector Space',
        ),
        models.Entity(
            type='theorem',
            members=[0],
            id=1,
            title='Basis Theorem',
            refs=[
                models.Reference(
                    target='vector space', kind='definition', relation='assumes'
                )
            ],
        ),
        models.Entity(
            type='definition',
            members=[0],
            id=2,
            title='Uncited Widget',  # never referenced -> no canonical -> no :REALIZES edge
        ),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await ensure_schema()
            await persist_nodes(stream, source)
            await persist_entities(overlay, source)
            await persist_references(overlay, source)
            await persist_realizes(overlay, source)
            await persist_realizes(overlay, source)  # idempotent re-run
            async with driver().session(database=database()) as session:
                # the "Vector Space" mention realizes the SAME canonical the theorem references,
                # so the citation resolves through the hub to where the concept is defined.
                realized = await one(
                    session,
                    "MATCH (m:Mention {title: 'Vector Space'})-[:REALIZES]->"
                    "(c:Canonical {type: 'definition'})"
                    "<-[:REFERENCES]-(:Entity {type: 'theorem'}) RETURN count(c) AS c",
                )
                # exactly one :REALIZES edge total: the uncited definition draws none (its title
                # matched no canonical), and the re-run did not duplicate.
                total = await one(
                    session,
                    'MATCH (:Mention)-[r:REALIZES]->(:Canonical) RETURN count(r) AS c',
                )
                assert realized['c'] == 1
                assert total['c'] == 1
        finally:
            async with driver().session(database=database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())


def test_persist_concepts_and_dependencies_build_the_prerequisite_graph():
    from kms.core import models
    from kms.graph.db import close_driver, database, driver
    from kms.graph.schema import ensure_schema
    from kms.graph.writer import (
        persist_concepts,
        persist_dependencies,
        persist_entities,
        persist_nodes,
        persist_procedures,
    )

    source = 'integration-test-book'
    stream = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content='…', id=0, segment_index=0
        )
    ]
    # A definition tagged "vector space", an eigenvalue definition whose proof step is tagged too,
    # and the prerequisite between their concepts.
    overlay = [
        models.Entity(
            type='definition',
            members=[0],
            id=0,
            title='Vector Space',
            concepts=['vector space', 'linear algebra'],
        ),
        models.Entity(
            type='definition',
            members=[0],
            id=1,
            title='Eigenvalue',
            concepts=['eigenvalue', 'linear algebra'],
            procedures=[
                models.Procedure(
                    type='proof',
                    contents=['Clear.'],
                    steps=[
                        models.BodySegment(
                            description='Assume not.',
                            action='assumption',
                            concepts=['proof by contradiction'],
                        )
                    ],
                )
            ],
        ),
    ]
    dependencies = [
        models.Dependency(
            dependent='eigenvalue', prerequisite='vector space', support=2
        ),
        # names a concept nothing instantiates -> MATCHed, never minted -> no edge
        models.Dependency(dependent='eigenvalue', prerequisite='sheaf'),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        try:
            await ensure_schema()
            await persist_nodes(stream, source)
            await persist_entities(overlay, source)
            await persist_procedures(overlay, source)
            await persist_concepts(overlay, source)
            await persist_concepts(overlay, source)  # idempotent re-run
            await persist_dependencies(dependencies)
            await persist_dependencies(dependencies)  # idempotent re-run
            async with driver().session(database=database()) as session:
                # the shared "linear algebra" tag converges on ONE concept both entities instance
                shared = await one(
                    session,
                    "MATCH (:Entity)-[:INSTANCE_OF]->(c:Concept {name: 'linear algebra'}) "
                    'RETURN count(c) AS c',
                )
                # a procedure step instances its own concept, the procedural half of the same axis
                stepwise = await one(
                    session,
                    "MATCH (:Event)-[:INSTANCE_OF]->(:Concept {name: 'proof by contradiction'}) "
                    'RETURN count(*) AS c',
                )
                # exactly one prerequisite edge: the sheaf dependency found no concept to attach to
                edges = await one(
                    session,
                    'MATCH (:Concept)-[d:DEPENDS_ON]->(:Concept) '
                    'RETURN count(d) AS c, collect(d.support)[0] AS s',
                )
                assert (
                    shared['c'] == 2
                )  # converged, and the re-run did not duplicate
                assert stepwise['c'] == 1
                assert edges['c'] == 1 and edges['s'] == 2
        finally:
            async with driver().session(database=database()) as session:
                await session.run(
                    'MATCH (n) DETACH DELETE n'
                )  # test DB: clear the graph
            await close_driver()

    asyncio.run(scenario())
