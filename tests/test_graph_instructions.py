"""Instruction-layer graph mapping and the :GOVERNS edges.

Pure mapping plus the instruction write, which runs against a fake driver —
the Cypher is asserted, nothing is sent anywhere.
"""

import asyncio

from kms.core import models
from kms.graph import instructions, nodes, statements, writer


def _instruction(node_id=0, members=None):
    return models.Instruction(
        node_id=node_id,
        text='In the following exercises, simplify.',
        directive='simplify',
        members=members if members is not None else [1, 2],
    )


def test_instruction_uuid_is_deterministic():
    assert instructions.instruction_uuid('ea2e.pdf', 7) == (
        instructions.instruction_uuid('ea2e.pdf', 7)
    )


def test_instruction_uuid_distinguishes_lead_in_and_source():
    assert instructions.instruction_uuid('ea2e.pdf', 7) != (
        instructions.instruction_uuid('ea2e.pdf', 8)
    )
    assert instructions.instruction_uuid('ea2e.pdf', 7) != (
        instructions.instruction_uuid('other.pdf', 7)
    )


def test_instruction_uuids_are_disjoint_from_other_tiers():
    key = instructions.instruction_uuid('book.pdf', 7)
    assert key != nodes.node_uuid('book.pdf', 7)
    assert key != statements.statement_uuid('book.pdf', [7])
    assert key != nodes.source_uuid('book.pdf')


def test_properties_keep_page_text_and_directive_apart():
    props = instructions.instruction_properties(_instruction(), 'ea2e.pdf')
    # The page's sentence and the model's imperative are different fields:
    # only one of them is something the document actually says.
    assert props['text'] == 'In the following exercises, simplify.'
    assert props['directive'] == 'simplify'
    assert props['index'] == 0


def test_properties_omit_an_absent_directive():
    hub = models.Instruction(node_id=0, text='Do these.', directive=None)
    props = instructions.instruction_properties(hub, 'ea2e.pdf')
    assert 'directive' not in props
    assert props['text'] == 'Do these.'


def test_governs_pairs_one_per_governed_node():
    pairs = instructions.governs_pairs(
        [_instruction(node_id=0, members=[1, 2, 3])], 'ea2e.pdf'
    )
    assert len(pairs) == 3
    assert {pair['node'] for pair in pairs} == {
        nodes.node_uuid('ea2e.pdf', member) for member in (1, 2, 3)
    }
    assert {pair['instruction'] for pair in pairs} == {
        instructions.instruction_uuid('ea2e.pdf', 0)
    }


def test_governs_pairs_empty_without_members():
    pairs = instructions.governs_pairs(
        [_instruction(node_id=0, members=[])], 'ea2e.pdf'
    )
    assert pairs == []


class _FakeSession:
    def __init__(self, log):
        self._log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self._log.append((query, params))
        return None


class _FakeDriver:
    def __init__(self):
        self.queries = []

    def session(self, **_):
        return _FakeSession(self.queries)


def test_persist_instructions_writes_hubs_then_governs_edges(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: fake)
    monkeypatch.setattr(writer, 'database', lambda: 'neo4j')

    asyncio.run(
        writer.persist_instructions(
            [_instruction(node_id=0, members=[1, 2])], 'ea2e.pdf'
        )
    )

    assert len(fake.queries) == 2
    hub_query, hub_params = fake.queries[0]
    assert f'MERGE (i:{instructions.INSTRUCTION_LABEL}' in hub_query
    assert len(hub_params['rows']) == 1

    edge_query, edge_params = fake.queries[1]
    # The edge runs from the hub outward, not from the node.
    assert 'MERGE (i)-[:GOVERNS]->(n)' in edge_query
    assert len(edge_params['pairs']) == 2


def test_persist_instructions_is_a_noop_when_empty(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(writer, 'driver', lambda: fake)
    asyncio.run(writer.persist_instructions([], 'ea2e.pdf'))
    assert fake.queries == []
