"""Variable-layer graph mapping and the :HAS_VARIABLE attachment edges.

Pure mapping plus the variable write, which runs against a fake driver — the
Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import nodes, variables, writer


def _variable(symbol='x', meaning='unknown'):
    return models.Variable(
        symbol=symbol,
        meaning=meaning,
        kind='variable',
    )


def _channel():
    # (node_id, [Variable]) pairs: one variable on node 1, one on node 2,
    # none on node 3.
    return [
        (1, [_variable(symbol='x')]),
        (2, [_variable(symbol='y')]),
    ]


def test_variable_uuid_is_deterministic():
    assert variables.variable_uuid('hefferon.pdf', 7, 'x') == (
        variables.variable_uuid('hefferon.pdf', 7, 'x')
    )


def test_variable_uuid_distinguishes_definitional_meanings():
    # "$X$ is a set", "$X$ is the set containing 0", "$X$ is the set
    # containing a and b" — one block, one symbol, three definitional
    # bindings. Without the meaning in the key all three MERGE onto the
    # same vertex and the last write wins.
    assert variables.variable_uuid('book.pdf', 7, '$X$', meaning='a set') != (
        variables.variable_uuid(
            'book.pdf', 7, '$X$', meaning='the set containing 0'
        )
    )
    assert variables.variable_uuid(
        'book.pdf', 7, '$X$', meaning='the set containing 0'
    ) != variables.variable_uuid(
        'book.pdf', 7, '$X$', meaning='the set containing a and b'
    )


def test_variable_uuid_distinguishes_bound_values():
    # "$-m$ when ⓐ $m = 3$ ⓑ $m = -3$" — one block, one symbol, two bindings.
    # Without the value in the key both MERGE onto the same vertex and the
    # second binding silently overwrites the first.
    assert variables.variable_uuid('ea2e.pdf', 7, 'm', '3') != (
        variables.variable_uuid('ea2e.pdf', 7, 'm', '-3')
    )


def test_variable_uuid_mixes_value_and_meaning_into_one_key():
    # A substitutional and a definitional binding of the same symbol in one
    # node differ even when only one of value/meaning is set — both parts
    # are always part of the key.
    assert variables.variable_uuid('book.pdf', 7, 'x', value='6') != (
        variables.variable_uuid('book.pdf', 7, 'x', meaning='the unknown')
    )


def test_variable_properties_carry_the_bound_value():
    props = variables.variable_properties(
        models.Variable(
            symbol='x',
            meaning='the variable being evaluated',
            kind='variable',
            value='6',
        ),
        'ea2e.pdf',
        3,
    )
    assert props['value'] == '6'
    assert props['symbol'] == 'x'


def test_variable_properties_omit_absent_value():
    props = variables.variable_properties(
        models.Variable(
            symbol='b', meaning='the cost of the blouse', kind='variable'
        ),
        'ea2e.pdf',
        3,
    )
    assert 'value' not in props


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
        'book.pdf', 4, 'alpha', meaning='learning rate'
    )
    assert props['source'] == nodes.source_uuid('book.pdf')
    assert props['symbol'] == 'alpha'
    assert props['meaning'] == 'learning rate'
    assert props['kind'] == 'variable'


def test_has_variable_pairs_hang_variables_off_their_node():
    pairs = variables.has_variable_pairs(_channel(), 'book.pdf')
    assert len(pairs) == 2
    assert pairs[0] == {
        'variable': variables.variable_uuid(
            'book.pdf', 1, 'x', meaning='unknown'
        ),
        'container': nodes.node_uuid('book.pdf', 1),
    }
    assert pairs[1] == {
        'variable': variables.variable_uuid(
            'book.pdf', 2, 'y', meaning='unknown'
        ),
        'container': nodes.node_uuid('book.pdf', 2),
    }


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


def test_persist_variables_writes_node_edges():
    driver = _FakeDriver()

    asyncio.run(
        writer.persist_variables(
            _channel(),
            'book.pdf',
            session_factory=lambda: driver.session(database='neo4j'),
        )
    )

    queries = [query for query, _ in driver.log]
    # Every variable hangs off :Node — a single label, no bucketing.
    assert any(
        '(n:Node {uuid: pair.container})' in query
        and '(n)-[:HAS_VARIABLE]->(v)' in query
        for query in queries
    )
    # No :Equation label is ever matched — the tier is gone.
    assert not any(':Equation' in query for query in queries)
    # No label-free MATCH that would scan across tiers.
    assert 'MATCH (a {uuid:' not in ' '.join(queries)
