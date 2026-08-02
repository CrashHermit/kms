"""Statement-overlay graph mapping, the pure provenance chain, and the
:MEMBER_OF edges.

Pure mapping plus the chain/overlay writes, which run against a fake driver —
the Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import nodes, statements, writer


def test_statement_uuid_is_deterministic():
    assert statements.statement_uuid('hefferon.pdf', [7]) == (
        statements.statement_uuid('hefferon.pdf', [7])
    )


def test_statement_uuid_distinguishes_block_and_source():
    assert statements.statement_uuid('hefferon.pdf', [7]) != (
        statements.statement_uuid('hefferon.pdf', [8])
    )
    assert statements.statement_uuid('hefferon.pdf', [7]) != (
        statements.statement_uuid('lebl.pdf', [7])
    )


def test_statement_uuid_covers_the_whole_block():
    # The identity is the whole block id set: a single-node block and a
    # multi-node block that starts at the same node never collide.
    assert statements.statement_uuid('book.pdf', [7]) != (
        statements.statement_uuid('book.pdf', [7, 8])
    )


def test_statement_uuids_are_disjoint_from_node_uuids():
    # A statement and its first member node name the same PLACE but are two
    # vertices in two tiers. They used to share a key, which was only
    # survivable while they were literally one fused `:Node:Statement` vertex.
    assert statements.statement_uuid('book.pdf', [7]) != nodes.node_uuid(
        'book.pdf', 7
    )


def test_statement_properties_carry_uuid_and_provenance_only():
    # A statement hub carries no text — the raw blocks carry it.
    statement = models.Statement(block=[4, 5], members=[4, 5])
    props = statements.statement_properties(statement, 'book.pdf')
    assert props['uuid'] == statements.statement_uuid('book.pdf', [4, 5])
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert 'content' not in props


def _stream():
    return [
        models.ParagraphNode(content='prose', id=0),
        models.ParagraphNode(content='Theorem 2.1.', id=1),
        models.ParagraphNode(content='Proof. ...', id=2),
        models.ParagraphNode(content='more prose', id=3),
    ]


def test_chain_is_the_pure_node_stream_in_document_order():
    # The provenance chain is the verbatim stream: even nodes absorbed into a
    # statement stay in it — statements are not chain elements.
    chain = writer._chain_nodes(_stream(), 'book.pdf')
    assert chain == [
        nodes.node_uuid('book.pdf', 0),
        nodes.node_uuid('book.pdf', 1),
        nodes.node_uuid('book.pdf', 2),
        nodes.node_uuid('book.pdf', 3),
    ]


def test_chain_pairs_thread_every_consecutive_node_pair():
    chain = writer._chain_nodes(_stream(), 'book.pdf')
    pairs = writer._chain_pairs(chain)
    assert pairs[0] == {
        'from': nodes.node_uuid('book.pdf', 0),
        'to': nodes.node_uuid('book.pdf', 1),
    }
    assert len(pairs) == 3


def test_an_empty_stream_has_no_chain():
    assert writer._chain_nodes([], 'book.pdf') == []
    assert writer._chain_pairs([]) == []


def test_statement_member_pairs_link_every_member_node():
    statement = models.Statement(block=[1, 2], members=[1, 2])
    pairs = statements.statement_member_pairs([statement], 'book.pdf')
    assert pairs == [
        {
            'node': nodes.node_uuid('book.pdf', 1),
            'statement': statements.statement_uuid('book.pdf', [1, 2]),
        },
        {
            'node': nodes.node_uuid('book.pdf', 2),
            'statement': statements.statement_uuid('book.pdf', [1, 2]),
        },
    ]


def test_statement_member_pairs_are_empty_without_statements():
    assert statements.statement_member_pairs([], 'book.pdf') == []


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


def test_persist_chain_writes_head_and_next_over_pure_nodes():
    driver = _FakeDriver()
    # The statement overlay is irrelevant to the chain: its members stay in
    # the verbatim stream.
    asyncio.run(
        writer.persist_chain(
            _stream(),
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    assert any(
        '(s:Source {uuid: $source})' in query
        and '(n:Node {uuid: $head})' in query
        and '(s)-[:HEAD]->(n)' in query
        for query in queries
    )
    assert any(
        '(a:Node {uuid: pair.from})' in query
        and '(b:Node {uuid: pair.to})' in query
        and '(a)-[:NEXT]->(b)' in query
        for query in queries
    )
    # The chain never mentions the statement tier: statements hang off their
    # member nodes via :MEMBER_OF, they are not chain elements. A
    # label-free `MATCH (a {uuid: ...})` would scan every vertex in the
    # database.
    assert 'Statement' not in ' '.join(queries)
    assert 'MATCH (a {uuid:' not in ' '.join(queries)


def test_persist_statements_writes_member_edges_from_every_member():
    driver = _FakeDriver()
    statement = models.Statement(block=[1, 2], members=[1, 2])

    asyncio.run(
        writer.persist_statements(
            [statement],
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    assert any(
        '(n:Node {uuid: pair.node})' in query
        and '(s:Statement {uuid: pair.statement})' in query
        and '(n)-[:MEMBER_OF]->(s)' in query
        for query in queries
    )
    # The membership link is real graph structure: one edge per member node,
    # absorbed ones included.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
