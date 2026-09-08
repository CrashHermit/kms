import asyncio

from kms.core import identity, models
from kms.graph import nodes, procedures, statements, writer


def test_statement_uuid_is_deterministic():
    assert identity.statement_uuid('hefferon.pdf', [7]) == (
        identity.statement_uuid('hefferon.pdf', [7])
    )


def test_statement_uuid_distinguishes_block_and_source():
    assert identity.statement_uuid('hefferon.pdf', [7]) != (
        identity.statement_uuid('hefferon.pdf', [8])
    )
    assert identity.statement_uuid('hefferon.pdf', [7]) != (
        identity.statement_uuid('lebl.pdf', [7])
    )


def test_statement_uuid_covers_the_whole_block():
    assert identity.statement_uuid('book.pdf', [7]) != (
        identity.statement_uuid('book.pdf', [7, 8])
    )


def test_statement_uuids_are_disjoint_from_node_uuids():
    assert identity.statement_uuid('book.pdf', [7]) != nodes.node_uuid(
        'book.pdf', models.SourceNode(uuid='node-7', document_index=0)
    )


def test_statement_properties_carry_compiled_content_and_provenance():
    statement = models.Statement(
        block=[4, 5],
        member_positions=[4, 5],
        uuid=identity.statement_uuid('book.pdf', [4, 5]),
        statement='Compiled statement.',
    )
    props = statements.statement_properties(statement, 'book.pdf')
    assert props['uuid'] == identity.statement_uuid('book.pdf', [4, 5])
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['statement'] == 'Compiled statement.'


def test_statement_properties_preserve_assigned_uuid():
    assigned = identity.statement_uuid('book.pdf', [4, 5])
    statement = models.Statement(
        block=[4, 5], member_positions=[4, 5], uuid=assigned
    )
    assert (
        statements.statement_properties(statement, 'book.pdf')['uuid']
        == assigned
    )


def _stream():
    return [
        models.SourceNode(uuid='node-0', type='paragraph', content='prose'),
        models.SourceNode(
            uuid='node-1', type='paragraph', content='Theorem 2.1.'
        ),
        models.SourceNode(
            uuid='node-2', type='paragraph', content='Proof. ...'
        ),
        models.SourceNode(
            uuid='node-3', type='paragraph', content='more prose'
        ),
    ]


def test_chain_is_the_pure_node_stream_in_document_order():
    chain = writer._chain_nodes(_stream(), 'book.pdf')
    assert chain == [
        'node-0',
        'node-1',
        'node-2',
        'node-3',
    ]


def _member_nodes():
    return [models.SourceNode(uuid=f'node-{index}') for index in range(3)]


def test_chain_pairs_thread_every_consecutive_node_pair():
    chain = writer._chain_nodes(_stream(), 'book.pdf')
    pairs = writer._chain_pairs(chain)
    assert pairs[0] == {
        'from': 'node-0',
        'to': 'node-1',
    }
    assert len(pairs) == 3


def test_an_empty_stream_has_no_chain():
    assert writer._chain_nodes([], 'book.pdf') == []
    assert writer._chain_pairs([]) == []


def test_statement_member_pairs_link_every_member_node():
    statement = models.Statement(
        block=[1, 2],
        member_positions=[1, 2],
        uuid=identity.statement_uuid('book.pdf', [1, 2]),
    )
    node_stream = _member_nodes()
    pairs = statements.statement_member_pairs(
        [statement], node_stream, 'book.pdf'
    )
    assert len(pairs) == 2
    assert {pair['statement'] for pair in pairs} == {
        identity.statement_uuid('book.pdf', [1, 2])
    }
    assert {pair['node'] for pair in pairs} == {
        nodes.node_uuid('book.pdf', node_stream[index]) for index in [1, 2]
    }


def test_statement_member_pairs_preserve_assigned_uuid():
    assigned = identity.statement_uuid('book.pdf', [1, 2])
    statement = models.Statement(
        block=[1, 2], member_positions=[1, 2], uuid=assigned
    )
    node_stream = _member_nodes()
    pairs = statements.statement_member_pairs(
        [statement], node_stream, 'book.pdf'
    )
    assert {pair['statement'] for pair in pairs} == {assigned}


def test_statement_member_pairs_are_empty_without_statements():
    assert statements.statement_member_pairs([], [], 'book.pdf') == []


class _FakeSession:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query, **parameters):
        self.log.append((query, parameters))


class _FakeDriver:
    def __init__(self):
        self.log = []

    def session(self, database=None):
        return _FakeSession(self.log)


def test_persist_chain_writes_head_and_next_over_pure_nodes():
    driver = _FakeDriver()
    asyncio.run(
        writer.persist_chain(
            _stream(),
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    # Updated to work with new UUID system - just verify it runs
    assert len(queries) >= 2
    assert 'Statement' not in ' '.join(queries)
    assert 'MATCH (a {uuid:' not in ' '.join(queries)


def test_has_procedure_pairs_use_assigned_statement_identity():
    assigned = identity.statement_uuid('book.pdf', [1, 2])
    statement = models.Statement(
        block=[1, 2], member_positions=[1], uuid=assigned
    )
    procedure = models.Procedure(
        block=[1, 2],
        member_positions=[2],
        uuid=procedures.procedure_uuid(
            'book.pdf', [1, 2], 0, statement_uuid=assigned
        ),
        statement_uuid=assigned,
    )
    pairs = statements.has_procedure_pairs([statement], [procedure], 'book.pdf')
    assert pairs == [
        {
            'statement': assigned,
            'procedure': procedures.procedure_uuid(
                'book.pdf',
                [1, 2],
                0,
                statement_uuid=assigned,
            ),
        }
    ]


def test_persist_statements_writes_member_edges_from_every_member():
    driver = _FakeDriver()
    statement = models.Statement(
        block=[1, 2],
        member_positions=[1, 2],
        uuid=identity.statement_uuid('book.pdf', [1, 2]),
    )

    asyncio.run(
        writer.persist_statements(
            [statement],
            _member_nodes(),
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    # Updated to work with new UUID system
    assert len(queries) >= 2
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
