"""Variable-layer graph mapping and the :HAS_VARIABLE attachment edges.

Pure mapping plus the variable write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import equations, nodes, variables, writer


def _variable(symbol='x', meaning='unknown', equation_index=None):
    return models.Variable(
        symbol=symbol,
        meaning=meaning,
        kind='variable',
        equation_index=equation_index,
    )


def _channel():
    # (node_id, [Variable]) pairs: two variables on node 1 (one free, one
    # bound to equation 0), one on node 2, none on node 3.
    return [
        (1, [
            _variable(symbol='x'),
            _variable(symbol='E', equation_index=0),
        ]),
        (2, [_variable(symbol='y')]),
    ]


def test_variable_uuid_is_deterministic():
    assert variables.variable_uuid('hefferon.pdf', 7, 'x') == (
        variables.variable_uuid('hefferon.pdf', 7, 'x')
    )


def test_variable_uuid_distinguishes_node_and_symbol():
    assert variables.variable_uuid('hefferon.pdf', 7, 'x') != (
        variables.variable_uuid('hefferon.pdf', 8, 'x')
    )
    assert variables.variable_uuid('hefferon.pdf', 7, 'x') != (
        variables.variable_uuid('hefferon.pdf', 7, 'y')
    )


def test_variable_uuids_are_disjoint_from_other_tiers():
    var = variables.variable_uuid('book.pdf', 7, 'x')
    assert var != nodes.node_uuid('book.pdf', 7)


def test_variable_properties_carry_symbol_and_provenance():
    props = variables.variable_properties(
        _variable(symbol='alpha', meaning='learning rate'),
        'book.pdf',
        4,
    )
    assert props['uuid'] == variables.variable_uuid(
        'book.pdf', 4, 'alpha'
    )
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['symbol'] == 'alpha'
    assert props['meaning'] == 'learning rate'
    assert props['kind'] == 'variable'


def test_has_variable_pairs_hang_free_variables_off_the_node():
    pairs = variables.has_variable_pairs(_channel(), 'book.pdf')
    node_pairs = [p for p in pairs if p['container_label'] == 'node']
    assert len(node_pairs) == 2
    assert node_pairs[0]['container'] == nodes.node_uuid('book.pdf', 1)
    assert node_pairs[1]['container'] == nodes.node_uuid('book.pdf', 2)


def test_has_variable_pairs_hang_equation_variables_off_the_equation():
    pairs = variables.has_variable_pairs(_channel(), 'book.pdf')
    equation_pairs = [p for p in pairs if p['container_label'] == 'equation']
    assert len(equation_pairs) == 1
    assert equation_pairs[0]['container'] == equations.equation_uuid(
        'book.pdf', 1, 0
    )
    assert equation_pairs[0]['variable'] == variables.variable_uuid(
        'book.pdf', 1, 'E'
    )


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


def test_persist_variables_writes_node_and_equation_edges(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: driver)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')

    asyncio.run(writer.persist_variables(_channel(), 'book.pdf'))

    queries = [query for query, _ in driver.log]
    # Free variables hang off :Node.
    assert any(
        '(n:Node {uuid: pair.container})' in query
        and '(n)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    # Equation-bound variables hang off :Equation.
    assert any(
        '(e:Equation {uuid: pair.container})' in query
        and '(e)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    # No label-free MATCH that would scan across tiers.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
