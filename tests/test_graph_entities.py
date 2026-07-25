"""models.Entity-overlay graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies identity is stable/deterministic, that a core.models.Entity maps onto the expected
Neo4j property shape, and that persist_entities issues the right queries/params via a fake session."""

import asyncio

from kms.core import models
from kms.graph.entities import entity_properties, entity_uuid
from kms.graph.nodes import node_uuid, source_uuid
from kms.graph.writer import entity_rows, member_pairs, persist_entities

_OVERLAY = [
    models.Entity(type='definition', members=[0], id=0, title='Group'),
    models.Entity(type='theorem', members=[1, 2], id=1, number='2.1'),
    models.Entity(type='exercise', members=[3], id=2, instruction='compute'),
]


# --- identity ---


def test_entity_uuid_is_deterministic_and_distinct_by_id_and_source():
    assert entity_uuid('book.pdf', 3) == entity_uuid('book.pdf', 3)
    assert entity_uuid('book.pdf', 3) != entity_uuid('book.pdf', 4)
    assert entity_uuid('book.pdf', 3) != entity_uuid('other.pdf', 3)


def test_entity_uuid_is_disjoint_from_node_uuid_for_the_same_index():
    # entity#3 and node#3 must not collide — they share a uuid namespace but different keys.
    assert entity_uuid('book.pdf', 3) != node_uuid('book.pdf', 3)


# --- properties ---


def test_entity_properties_map_identity_source_and_scalars():
    props = entity_properties(_OVERLAY[0], 'book.pdf')
    assert props['uuid'] == entity_uuid('book.pdf', 0)
    assert props['source'] == source_uuid('book.pdf')
    assert props['type'] == 'definition'
    assert props['title'] == 'Group'


def test_type_is_an_open_induced_string_not_a_closed_vocabulary():
    # a non-math genre must map through untouched — no enum, no validation
    props = entity_properties(
        models.Entity(type='law', members=[0], id=0), 'book.pdf'
    )
    assert props['type'] == 'law'


def test_entity_properties_omit_unset_attributes():
    props = entity_properties(
        models.Entity(type='exercise', members=[3], id=2), 'book.pdf'
    )
    for absent in ('label', 'number', 'title', 'instruction', 'contents'):
        assert absent not in props


def test_retired_automathkg_attributes_are_gone():
    props = entity_properties(_OVERLAY[1], 'book.pdf')
    for retired in ('field', 'bodylist', 'proofs', 'solutions', 'refs'):
        assert retired not in props


def test_entity_properties_keep_id_zero():
    props = entity_properties(
        models.Entity(type='definition', members=[0], id=0), 'book.pdf'
    )
    assert props['uuid'] == entity_uuid(
        'book.pdf', 0
    )  # a falsy-but-valid id is not dropped


def test_contents_is_a_native_string_array():
    entity = models.Entity(
        type='theorem', members=[1], id=1, contents=['Let n be prime.']
    )
    props = entity_properties(entity, 'book.pdf')
    assert props['contents'] == ['Let n be prime.']


def test_derivations_are_not_on_the_entity_they_reify_into_procedures():
    # procedures are the procedural layer (graph.procedures) — never entity props
    entity = models.Entity(
        type='theorem',
        members=[1],
        id=1,
        procedures=[models.Procedure(contents=['Clear.'], steps=['Clear.'])],
    )
    props = entity_properties(entity, 'book.pdf')
    assert 'procedures' not in props


# --- writer planning ---


def test_entity_rows_are_one_flat_batch_with_no_per_type_label():
    rows = entity_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 3
    assert {row['type'] for row in rows} == {
        'definition',
        'theorem',
        'exercise',
    }
    assert rows[1]['number'] == '2.1'


def test_member_pairs_are_one_per_entity_member():
    pairs = member_pairs(_OVERLAY, 'book.pdf')
    assert len(pairs) == 4  # 1 + 2 + 1 members across the overlay
    assert pairs[0] == {
        'entity': entity_uuid('book.pdf', 0),
        'node': node_uuid('book.pdf', 0),
    }
    assert pairs[-1] == {
        'entity': entity_uuid('book.pdf', 2),
        'node': node_uuid('book.pdf', 3),
    }


# --- persist_entities orchestration, via a fake session (no server) ---


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.calls.append((query, params))


class _FakeDriver:
    def __init__(self, calls):
        self.calls = calls

    def session(self, **kwargs):
        return _FakeSession(self.calls)


def test_persist_entities_writes_vertices_root_and_members(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_entities(_OVERLAY, 'book.pdf'))

    queries = [query for query, _ in calls]
    # one batched MERGE on the bare :Entity label — no per-type and no role labels
    assert any('MERGE (e:Entity' in query for query in queries)
    assert not any('SET e:Theorem' in query for query in queries)
    assert not any('SET e:Mention' in query for query in queries)
    # entities are rooted under their :Source via :HAS_ENTITY
    root = next(call for call in calls if ':HAS_ENTITY' in call[0])
    assert root[1]['src'] == source_uuid('book.pdf')
    assert set(root[1]['uuids']) == {
        entity_uuid('book.pdf', i) for i in range(3)
    }
    # members are linked via :DERIVED_FROM, one edge per (entity, member)
    members = next(call for call in calls if ':DERIVED_FROM' in call[0])
    assert len(members[1]['pairs']) == 4


def test_persist_entities_is_a_noop_for_an_empty_overlay(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_entities([], 'book.pdf'))
    assert calls == []  # nothing opened, nothing written
