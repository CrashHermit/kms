"""Statement-overlay graph mapping and the merged chain.

Pure mapping plus the chain write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import nodes, statements, writer


def test_statement_uuid_is_deterministic():
    assert statements.statement_uuid('hefferon.pdf', 7) == (
        statements.statement_uuid('hefferon.pdf', 7)
    )


def test_statement_uuid_distinguishes_id_and_source():
    assert statements.statement_uuid('hefferon.pdf', 7) != (
        statements.statement_uuid('hefferon.pdf', 8)
    )
    assert statements.statement_uuid('hefferon.pdf', 7) != (
        statements.statement_uuid('lebl.pdf', 7)
    )


def test_statement_uuids_are_disjoint_from_node_uuids():
    # A statement and its first member node name the same PLACE but are two
    # vertices in two tiers. They used to share a key, which was only
    # survivable while they were literally one fused `:Node:Statement` vertex.
    assert statements.statement_uuid('book.pdf', 7) != nodes.node_uuid(
        'book.pdf', 7
    )


def test_statement_properties_carry_content_and_provenance():
    statement = models.Statement(id=4, content='Theorem 2.1.', statement_of=[4, 5])
    props = statements.statement_properties(statement, 'book.pdf')
    assert props['uuid'] == statements.statement_uuid('book.pdf', 4)
    assert props['content'] == 'Theorem 2.1.'
    assert props['source'] == nodes.source_uuid('book.pdf')


def test_statement_properties_drop_empty_content():
    statement = models.Statement(id=4, statement_of=[4])
    assert 'content' not in statements.statement_properties(
        statement, 'book.pdf'
    )


def _stream():
    return [
        models.ParagraphNode(content='prose', id=0),
        models.ParagraphNode(content='Theorem 2.1.', id=1),
        models.ParagraphNode(content='Proof. ...', id=2),
        models.ParagraphNode(content='more prose', id=3),
    ]


def test_chain_slots_the_statement_in_and_skips_its_members():
    statement = models.Statement(id=1, statement_of=[1, 2])
    elements = writer._chain_elements(_stream(), [statement], 'book.pdf')
    assert [element['label'] for element in elements] == [
        'Node',
        'Statement',
        'Node',
    ]
    assert [element['uuid'] for element in elements] == [
        nodes.node_uuid('book.pdf', 0),
        statements.statement_uuid('book.pdf', 1),
        nodes.node_uuid('book.pdf', 3),
    ]


def test_chain_pairs_carry_both_endpoints_labels():
    # Cypher cannot parameterise a label, so persist_chain buckets on these —
    # a label-free MATCH would scan, and would break outright if two tiers ever
    # shared a uuid.
    statement = models.Statement(id=1, statement_of=[1, 2])
    pairs = writer._merged_chain(_stream(), [statement], 'book.pdf')
    assert [(pair['from_label'], pair['to_label']) for pair in pairs] == [
        ('Node', 'Statement'),
        ('Statement', 'Node'),
    ]


def test_head_is_the_first_element_with_its_label():
    statement = models.Statement(id=0, statement_of=[0, 1])
    head = writer._merged_head(_stream(), [statement], 'book.pdf')
    assert head == {
        'label': 'Statement',
        'uuid': statements.statement_uuid('book.pdf', 0),
    }


def test_head_of_an_overlay_free_stream_is_the_first_node():
    head = writer._merged_head(_stream(), [], 'book.pdf')
    assert head == {'label': 'Node', 'uuid': nodes.node_uuid('book.pdf', 0)}


def test_an_empty_stream_has_no_chain():
    assert writer._chain_elements([], [], 'book.pdf') == []
    assert writer._merged_head([], [], 'book.pdf') is None


class _FakeSession:
    """Records the Cypher it is handed instead of running it."""

    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query, **params):
        self.log.append((query, params))


class _FakeDriver:
    def __init__(self):
        self.log = []

    def session(self, database=None):
        return _FakeSession(self.log)


def test_persist_chain_matches_every_endpoint_by_label(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: driver)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')
    statement = models.Statement(id=1, statement_of=[1, 2])

    asyncio.run(writer.persist_chain(_stream(), [statement], 'book.pdf'))

    queries = [query for query, _ in driver.log]
    # The HEAD edge and both :NEXT buckets name the label they match on. A
    # label-free `MATCH (a {uuid: ...})` scans every vertex in the database and
    # would match across tiers outright if two ever shared a uuid.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
    assert any('(n:Node {uuid: $head})' in query for query in queries)
    assert any(
        '(a:Node {uuid: pair.from})' in query
        and '(b:Statement {uuid: pair.to})' in query
        for query in queries
    )
    assert any(
        '(a:Statement {uuid: pair.from})' in query
        and '(b:Node {uuid: pair.to})' in query
        for query in queries
    )
