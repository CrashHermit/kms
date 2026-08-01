"""Equation-layer graph mapping and the attachment edges.

Pure mapping plus the equation write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import (
    equations,
    nodes,
    procedures,
    statements,
    writer,
)


def _equation(latex='$$E = mc^2$$', name=None, domain=None):
    return models.Equation(latex=latex, name=name, domain=domain)


def _channel():
    # (unit_kind, block, [Equation]) triples: one statement unit (block
    # [1, 2]), one procedure unit (the SAME block — a different kind), and
    # one plain node unit (block [3]).
    return [
        (
            models.UNIT_STATEMENT,
            [1, 2],
            [
                _equation(name='mass-energy equivalence'),
                _equation(latex='$$V = IR$$', domain='circuit analysis'),
            ],
        ),
        (
            models.UNIT_PROCEDURE,
            [1, 2],
            [_equation(latex='$$a^2 = b^2 + c^2$$')],
        ),
        (models.UNIT_NODE, [3], [_equation(latex='$$y = mx + b$$')]),
    ]


def _procedures():
    return [models.Procedure(block=[1, 2], members=[1, 2])]


def test_equation_uuid_is_deterministic():
    assert equations.equation_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 0
    ) == equations.equation_uuid('hefferon.pdf', models.UNIT_STATEMENT, [7], 0)


def test_equation_uuid_distinguishes_kind_block_and_index():
    assert equations.equation_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 0
    ) != equations.equation_uuid('hefferon.pdf', models.UNIT_PROCEDURE, [7], 0)
    assert equations.equation_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 0
    ) != equations.equation_uuid('hefferon.pdf', models.UNIT_STATEMENT, [8], 0)
    assert equations.equation_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 0
    ) != equations.equation_uuid('hefferon.pdf', models.UNIT_STATEMENT, [7], 1)


def test_equation_uuids_are_disjoint_from_other_tiers():
    eq = equations.equation_uuid('book.pdf', models.UNIT_STATEMENT, [7], 0)
    assert eq != nodes.node_uuid('book.pdf', 7)
    assert eq != statements.statement_uuid('book.pdf', [7])


def test_equation_properties_carry_latex_and_provenance():
    props = equations.equation_properties(
        _equation(name='heat equation'),
        'book.pdf',
        models.UNIT_STATEMENT,
        [4],
        0,
    )
    assert props['uuid'] == equations.equation_uuid(
        'book.pdf', models.UNIT_STATEMENT, [4], 0
    )
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['latex'] == '$$E = mc^2$$'
    assert props['name'] == 'heat equation'


def test_equation_properties_drop_empty_fields():
    props = equations.equation_properties(
        _equation(), 'book.pdf', models.UNIT_STATEMENT, [4], 0
    )
    assert 'name' not in props
    assert 'domain' not in props


def test_equation_rows_flatten_across_units():
    rows = equations.equation_rows(_channel(), 'book.pdf')
    assert [row['uuid'] for row in rows] == [
        equations.equation_uuid('book.pdf', models.UNIT_STATEMENT, [1, 2], 0),
        equations.equation_uuid('book.pdf', models.UNIT_STATEMENT, [1, 2], 1),
        equations.equation_uuid('book.pdf', models.UNIT_PROCEDURE, [1, 2], 0),
        equations.equation_uuid('book.pdf', models.UNIT_NODE, [3], 0),
    ]


def test_equation_pairs_hang_statement_equations_off_the_statement():
    pairs = equations.equation_pairs(_channel(), _procedures(), 'book.pdf')
    statement_pairs = [p for p in pairs if p['container_label'] == 'statement']
    assert len(statement_pairs) == 2
    assert all(
        p['container'] == statements.statement_uuid('book.pdf', [1, 2])
        for p in statement_pairs
    )
    # The container is the statement hub, never a member node's uuid — those
    # nodes are invisible in the chain.
    assert all(
        p['container'] != nodes.node_uuid('book.pdf', 1)
        for p in statement_pairs
    )


def test_equation_pairs_hang_procedure_equations_off_the_procedure():
    pairs = equations.equation_pairs(_channel(), _procedures(), 'book.pdf')
    procedure_pairs = [p for p in pairs if p['container_label'] == 'procedure']
    assert len(procedure_pairs) == 1
    assert procedure_pairs[0]['container'] == procedures.procedure_uuid(
        'book.pdf', [1, 2], 0
    )


def test_equation_pairs_fall_back_to_the_plain_node():
    pairs = equations.equation_pairs(_channel(), _procedures(), 'book.pdf')
    node_pairs = [p for p in pairs if p['container_label'] == 'node']
    assert len(node_pairs) == 1
    assert node_pairs[0]['container'] == nodes.node_uuid('book.pdf', 3)


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


def test_persist_equations_matches_each_container_by_label(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: driver)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')

    asyncio.run(writer.persist_equations(_channel(), _procedures(), 'book.pdf'))

    queries = [query for query, _ in driver.log]
    # Statement-, procedure- and node-owned equations all MERGE the same
    # :HAS_EQUATION edge off their own tier — every unit is an equally valid
    # extraction source.
    assert any(
        '(s:Statement {uuid: pair.container})' in query
        and '(s)-[:HAS_EQUATION]->(e)' in query
        for query in queries
    )
    assert any(
        '(p:Procedure {uuid: pair.container})' in query
        and '(p)-[:HAS_EQUATION]->(e)' in query
        for query in queries
    )
    assert any(
        '(n:Node {uuid: pair.container})' in query
        and '(n)-[:HAS_EQUATION]->(e)' in query
        for query in queries
    )
    # A label-free `MATCH (a {uuid: ...})` would scan every vertex in the
    # database and would match across tiers outright if two ever shared a uuid.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
