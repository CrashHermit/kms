import asyncio

from kms.core import models
from kms.graph import nodes, procedures, writer


def _procedure(block=(1, 2), member_positions=(1, 2), statement_uuid=None):
    return models.Procedure(
        block=list(block),
        member_positions=list(member_positions),
        statement_uuid=statement_uuid,
        uuid=procedures.procedure_uuid(
            'book.pdf',
            list(block),
            0,
            statement_uuid=statement_uuid,
            member_positions=list(member_positions),
        ),
    )


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
    assert procedures.procedure_uuid('book.pdf', [7], 0) != (
        procedures.procedure_uuid('book.pdf', [7, 8], 0)
    )
def test_procedure_uuid_separates_source_and_generated_kinds():
    source_uuid = procedures.procedure_uuid(
        'book.pdf',
        [7],
        0,
        statement_uuid='statement-a',
        member_positions=[7],
    )
    generated_uuid = procedures.procedure_uuid(
        'book.pdf',
        [],
        0,
        statement_uuid='statement-a',
        kind=models.ProcedureKind.GENERATED,
    )
    assert source_uuid != generated_uuid
    assert generated_uuid == procedures.procedure_uuid(
        'book.pdf',
        [],
        0,
        statement_uuid='statement-a',
        kind=models.ProcedureKind.GENERATED,
    )


def test_procedure_uuid_distinguishes_statement_backed_procedures():
    assert procedures.procedure_uuid(
        'book.pdf', [], 0, statement_uuid='statement-a'
    ) != procedures.procedure_uuid(
        'book.pdf', [], 0, statement_uuid='statement-b'
    )


def test_statement_backed_procedure_persistence_uses_its_disambiguated_uuid():
    procedure = _procedure(statement_uuid='statement-a')
    expected = procedures.procedure_uuid(
        'book.pdf',
        [1, 2],
        0,
        statement_uuid='statement-a',
        member_positions=[1, 2],
    )
    assert procedures.procedure_properties('book.pdf', procedure)['uuid'] == (
        expected
    )


def test_procedure_properties_carry_content_provenance_and_kind():
    procedure = _procedure()
    procedure.procedure = 'Compiled procedure.'
    props = procedures.procedure_properties('book.pdf', procedure)
    assert props['uuid'] == procedures.procedure_uuid(
        'book.pdf', [1, 2], 0, member_positions=[1, 2]
    )
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['kind'] == 'source'
    assert props['procedure'] == 'Compiled procedure.'

def test_procedure_member_pairs_link_every_member_node():
    doc_nodes = [
        models.SourceNode(document_index=0, index=0),
        models.SourceNode(document_index=0, index=1),
        models.SourceNode(document_index=0, index=2),
    ]
    pairs = procedures.procedure_member_pairs(
        [_procedure()], 'book.pdf', doc_nodes
    )
    assert {pair['procedure'] for pair in pairs} == {
        procedures.procedure_uuid(
            'book.pdf', [1, 2], 0, member_positions=[1, 2]
        )
    }
    # Node UUIDs should be resolved from the actual node stream
    assert pairs[0]['node'] == nodes.node_uuid('book.pdf', doc_nodes[1])
    assert pairs[1]['node'] == nodes.node_uuid('book.pdf', doc_nodes[2])


def test_procedure_member_pairs_are_empty_without_procedures():
    assert procedures.procedure_member_pairs([], 'book.pdf', []) == []


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


def test_persist_procedures_points_each_member_at_the_procedure():
    driver = _FakeDriver()
    doc_nodes = [
        models.SourceNode(document_index=0, index=0),
        models.SourceNode(document_index=0, index=1),
        models.SourceNode(document_index=0, index=2),
    ]

    asyncio.run(
        writer.persist_procedures(
            [_procedure()],
            doc_nodes,
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
    assert 'HAS_PROCEDURE' not in ' '.join(queries)
    assert 'MATCH (a {uuid:' not in ' '.join(queries)