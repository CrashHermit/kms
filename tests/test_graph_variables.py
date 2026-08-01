"""Variable-layer graph mapping and the :HAS_VARIABLE attachment edges.

Pure mapping plus the variable write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import (
    equations,
    nodes,
    procedures,
    statements,
    variables,
    writer,
)


def _variable(symbol='x', meaning='unknown', equation_index=None):
    return models.Variable(
        symbol=symbol,
        meaning=meaning,
        kind='variable',
        equation_index=equation_index,
    )


def _channel():
    # (unit_kind, block, [Variable]) triples: one statement unit (block
    # [1, 2]), one procedure unit (the SAME block — a different kind), one
    # equation binding inside the statement, and one plain node unit.
    return [
        (
            models.UNIT_STATEMENT,
            [1, 2],
            [_variable(symbol='x'), _variable(symbol='E', equation_index=0)],
        ),
        (models.UNIT_PROCEDURE, [1, 2], [_variable(symbol='n')]),
        (models.UNIT_NODE, [3], [_variable(symbol='y')]),
    ]


def _procedures():
    return [models.Procedure(block=[1, 2], members=[1, 2])]


def test_variable_uuid_is_deterministic():
    assert variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'x'
    ) == variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'x'
    )


def test_variable_uuid_distinguishes_kind_block_and_symbol():
    assert variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'x'
    ) != variables.variable_uuid(
        'hefferon.pdf', models.UNIT_PROCEDURE, [7], 'x'
    )
    assert variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'x'
    ) != variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [8], 'x'
    )
    assert variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'x'
    ) != variables.variable_uuid(
        'hefferon.pdf', models.UNIT_STATEMENT, [7], 'y'
    )


def test_variable_uuids_are_disjoint_from_other_tiers():
    var = variables.variable_uuid('book.pdf', models.UNIT_STATEMENT, [7], 'x')
    assert var != nodes.node_uuid('book.pdf', 7)
    assert var != statements.statement_uuid('book.pdf', [7])


def test_variable_properties_carry_symbol_and_provenance():
    props = variables.variable_properties(
        _variable(symbol='alpha', meaning='learning rate'),
        'book.pdf',
        models.UNIT_STATEMENT,
        [4],
    )
    assert props['uuid'] == variables.variable_uuid(
        'book.pdf', models.UNIT_STATEMENT, [4], 'alpha'
    )
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['symbol'] == 'alpha'
    assert props['meaning'] == 'learning rate'
    assert props['kind'] == 'variable'


def test_has_variable_pairs_hang_statement_variables_off_the_statement():
    pairs = variables.has_variable_pairs(_channel(), _procedures(), 'book.pdf')
    statement_pairs = [p for p in pairs if p['container_label'] == 'statement']
    assert len(statement_pairs) == 1
    assert statement_pairs[0]['container'] == statements.statement_uuid(
        'book.pdf', [1, 2]
    )
    # The container is the statement hub, never a member node's uuid — those
    # nodes are invisible in the chain.
    assert statement_pairs[0]['container'] != nodes.node_uuid('book.pdf', 1)


def test_has_variable_pairs_hang_procedure_variables_off_the_procedure():
    pairs = variables.has_variable_pairs(_channel(), _procedures(), 'book.pdf')
    procedure_pairs = [p for p in pairs if p['container_label'] == 'procedure']
    assert len(procedure_pairs) == 1
    assert procedure_pairs[0]['container'] == procedures.procedure_uuid(
        'book.pdf', [1, 2], 0
    )


def test_has_variable_pairs_hang_equation_variables_off_the_equation():
    pairs = variables.has_variable_pairs(_channel(), _procedures(), 'book.pdf')
    equation_pairs = [p for p in pairs if p['container_label'] == 'equation']
    assert len(equation_pairs) == 1
    assert equation_pairs[0]['container'] == equations.equation_uuid(
        'book.pdf', models.UNIT_STATEMENT, [1, 2], 0
    )


def test_has_variable_pairs_fall_back_to_the_plain_node():
    pairs = variables.has_variable_pairs(_channel(), _procedures(), 'book.pdf')
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


def test_persist_variables_matches_each_container_by_label(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: driver)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')

    asyncio.run(writer.persist_variables(_channel(), _procedures(), 'book.pdf'))

    queries = [query for query, _ in driver.log]
    # Each container tier gets its own label-parameterised :HAS_VARIABLE edge.
    assert any(
        '(e:Equation {uuid: pair.container})' in query
        and '(e)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    assert any(
        '(s:Statement {uuid: pair.container})' in query
        and '(s)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    assert any(
        '(p:Procedure {uuid: pair.container})' in query
        and '(p)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    assert any(
        '(n:Node {uuid: pair.container})' in query
        and '(n)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    # A label-free `MATCH (a {uuid: ...})` would scan every vertex in the
    # database and would match across tiers outright if two ever shared a uuid.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
