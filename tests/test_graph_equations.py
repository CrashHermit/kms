"""Equation-layer graph mapping and the :HAS_EQUATION attachment edges.

Pure mapping plus the equation write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import equations, nodes, writer


def _equation(latex='$$E = mc^2$$', name=None, domain=None):
    return models.Equation(latex=latex, name=name, domain=domain)


def _channel():
    # (node_id, [Equation]) pairs: one equation on node 1, two on node 2,
    # none on node 3.
    return [
        (1, [_equation(name='mass-energy equivalence')]),
        (2, [
            _equation(latex='$$V = IR$$', domain='circuit analysis'),
            _equation(latex='$$a^2 = b^2 + c^2$$'),
        ]),
    ]


def test_equation_uuid_is_deterministic():
    assert equations.equation_uuid('hefferon.pdf', 7, 0) == (
        equations.equation_uuid('hefferon.pdf', 7, 0)
    )


def test_equation_uuid_distinguishes_node_and_index():
    assert equations.equation_uuid('hefferon.pdf', 7, 0) != (
        equations.equation_uuid('hefferon.pdf', 8, 0)
    )
    assert equations.equation_uuid('hefferon.pdf', 7, 0) != (
        equations.equation_uuid('hefferon.pdf', 7, 1)
    )


def test_equation_uuids_are_disjoint_from_other_tiers():
    eq = equations.equation_uuid('book.pdf', 7, 0)
    assert eq != nodes.node_uuid('book.pdf', 7)


def test_equation_properties_carry_latex_and_provenance():
    props = equations.equation_properties(
        _equation(name='heat equation'), 'book.pdf', 4, 0
    )
    assert props['uuid'] == equations.equation_uuid('book.pdf', 4, 0)
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['latex'] == '$$E = mc^2$$'
    assert props['name'] == 'heat equation'


def test_equation_properties_drop_empty_fields():
    props = equations.equation_properties(
        _equation(), 'book.pdf', 4, 0
    )
    assert 'name' not in props
    assert 'domain' not in props


def test_equation_rows_flatten_across_nodes():
    rows = equations.equation_rows(_channel(), 'book.pdf')
    assert [row['uuid'] for row in rows] == [
        equations.equation_uuid('book.pdf', 1, 0),
        equations.equation_uuid('book.pdf', 2, 0),
        equations.equation_uuid('book.pdf', 2, 1),
    ]


def test_equation_pairs_hang_every_equation_off_its_node():
    pairs = equations.equation_pairs(_channel(), 'book.pdf')
    assert pairs == [
        {
            'node': nodes.node_uuid('book.pdf', 1),
            'equation': equations.equation_uuid('book.pdf', 1, 0),
        },
        {
            'node': nodes.node_uuid('book.pdf', 2),
            'equation': equations.equation_uuid('book.pdf', 2, 0),
        },
        {
            'node': nodes.node_uuid('book.pdf', 2),
            'equation': equations.equation_uuid('book.pdf', 2, 1),
        },
    ]
    # Every equation hangs off a :Node — no statement/procedure labels.
    assert all(
        'node' in pair and 'equation' in pair for pair in pairs
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


def test_persist_equations_writes_node_has_equation_edges(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: driver)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')

    asyncio.run(writer.persist_equations(_channel(), 'book.pdf'))

    queries = [query for query, _ in driver.log]
    # All equations hang off :Node — a single label, no bucketing.
    assert any(
        '(n:Node {uuid: pair.node})' in query
        and '(e:Equation {uuid: pair.equation})' in query
        and '(n)-[:HAS_EQUATION]->(e)' in query
        for query in queries
    )
    # No label-free MATCH that would scan across tiers.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
