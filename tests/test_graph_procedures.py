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
    assert procedures.procedure_uuid('book.pdf', [7], 0) != (
        procedures.procedure_uuid('book.pdf', [7, 8], 0)
    )


def test_procedure_uuid_distinguishes_statement_backed_procedures():
    assert procedures.procedure_uuid(
        'book.pdf', [], 0, statement_uuid='statement-a'
    ) != procedures.procedure_uuid(
        'book.pdf', [], 0, statement_uuid='statement-b'
    )


def test_statement_backed_procedure_persistence_uses_its_disambiguated_uuid():
    procedure = _procedure()
    procedure.statement_uuid = 'statement-a'
    expected = procedures.procedure_uuid(
        'book.pdf', [1, 2], 0, statement_uuid='statement-a'
    )

    assert (
        procedures.procedure_properties('book.pdf', procedure)['uuid']
        == expected
    )
    assert (
        procedures.procedure_member_pairs([procedure], 'book.pdf')[0][
            'procedure'
        ]
        == expected
    )


def test_procedure_properties_carry_index_and_provenance_only():
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


def test_existing_step_rows_use_existing_procedure_uuid():
    steps = [models.Step(text='Set up.', index=0)]

    rows = procedures.existing_step_rows('book.pdf', 'procedure-a', steps)

    assert rows[0]['uuid'] == procedures.step_uuid('book.pdf', 'procedure-a', 0)


def test_existing_edges_use_existing_procedure_uuid():
    steps = [
        models.Step(text='Set up.', index=0),
        models.Step(text='Conclude.', index=1),
    ]

    assert procedures.existing_first_pairs(
        'book.pdf', 'procedure-a', steps
    ) == [
        {
            'procedure': 'procedure-a',
            'step': procedures.step_uuid('book.pdf', 'procedure-a', 0),
        }
    ]
    assert procedures.existing_then_pairs('book.pdf', 'procedure-a', steps) == [
        {
            'from': procedures.step_uuid('book.pdf', 'procedure-a', 0),
            'to': procedures.step_uuid('book.pdf', 'procedure-a', 1),
        }
    ]


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


def test_persist_existing_procedure_steps_uses_existing_uuid():
    driver = _FakeDriver()
    steps = [models.Step(text='Set up.', index=0)]

    asyncio.run(
        writer.persist_procedure_steps(
            'procedure-a',
            steps,
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    step_query, step_parameters = driver.log[0]
    assert 'MERGE (s:Step {uuid: row.uuid})' in step_query
    assert step_parameters['rows'][0]['uuid'] == procedures.step_uuid(
        'book.pdf', 'procedure-a', 0
    )


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
    assert 'HAS_PROCEDURE' not in ' '.join(queries)
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
