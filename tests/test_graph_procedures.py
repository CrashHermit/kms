"""Procedural-layer graph mapping and the :MEMBER_OF attachment.

Pure mapping plus the procedure write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import nodes, procedures, writer


def _procedure(block=(1, 2), members=(1, 2)):
    return models.Procedure(block=list(block), members=list(members))


def test_procedure_uuid_is_deterministic():
    assert procedures.procedure_uuid('hefferon.pdf', [7], 0) == (
        procedures.procedure_uuid('hefferon.pdf', [7], 0)
    )


def test_procedure_uuid_distinguishes_block_and_index():
    assert procedures.procedure_uuid('hefferon.pdf', [7], 0) != (
        procedures.procedure_uuid('hefferon.pdf', [8], 0)
    )
    assert procedures.procedure_uuid('hefferon.pdf', [7], 0) != (
        procedures.procedure_uuid('hefferon.pdf', [7], 1)
    )


def test_procedure_uuid_covers_the_whole_block():
    # A single-node block and a multi-node block starting at the same node
    # never collide.
    assert procedures.procedure_uuid('book.pdf', [7], 0) != (
        procedures.procedure_uuid('book.pdf', [7, 8], 0)
    )


def test_procedure_properties_carry_index_and_provenance_only():
    # A procedure hub carries no text — the raw blocks carry it.
    props = procedures.procedure_properties('book.pdf', _procedure())
    assert props['uuid'] == procedures.procedure_uuid('book.pdf', [1, 2], 0)
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert 'content' not in props


def test_procedure_member_pairs_link_every_member_node():
    pairs = procedures.procedure_member_pairs([_procedure()], 'book.pdf')
    assert pairs == [
        {
            'node': nodes.node_uuid('book.pdf', 1),
            'procedure': procedures.procedure_uuid('book.pdf', [1, 2], 0),
        },
        {
            'node': nodes.node_uuid('book.pdf', 2),
            'procedure': procedures.procedure_uuid('book.pdf', [1, 2], 0),
        },
    ]


def test_procedure_member_pairs_are_empty_without_procedures():
    assert procedures.procedure_member_pairs([], 'book.pdf') == []


class _FakeSession:
    """Records the Cypher it is handed instead of running it."""

    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query, **parameters):
        self.log.append((query, parameters))


class _FakeDriver:
    """Hands out recording sessions in place of a Neo4j driver."""

    def __init__(self):
        self.log = []

    def session(self, database=None):
        return _FakeSession(self.log)


def test_persist_procedures_points_each_member_at_the_procedure():
    driver = _FakeDriver()

    asyncio.run(
        writer.persist_procedures(
            [_procedure()],
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    assert any(
        '(n:Node {uuid: pair.node})' in query
        and '(p:Procedure {uuid: pair.procedure})' in query
        and '(n)-[r:MEMBER_OF]->(p)' in query
        for query in queries
    )
    # A procedure hub is independent: no statement↔procedure edge is written
    # at the grouping phase — that relationship is parked for the semantic
    # tier. A label-free `MATCH (a {uuid: ...})` would scan every vertex in
    # the database.
    assert 'HAS_PROCEDURE' not in ' '.join(queries)
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
