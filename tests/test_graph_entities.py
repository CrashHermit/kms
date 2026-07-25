"""models.Entity-overlay graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies identity is stable/deterministic, that a core.models.Entity maps onto the expected
Neo4j property shape, and that persist_entities issues the right queries/params via a fake session."""

import asyncio
import json

from kms.core import models
from kms.graph.entities import entity_properties, entity_uuid
from kms.graph.nodes import node_uuid, source_uuid
from kms.graph.writer import entity_rows, member_pairs, persist_entities

_OVERLAY = [
    models.Entity(type='definition', members=[0], id=0, title='Group'),
    models.Entity(type='theorem', members=[1, 2], id=1, number='2.1'),
    models.Entity(type='problem', members=[3], id=2, instruction='compute'),
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


def test_entity_type_is_an_open_property_not_a_closed_enum():
    # a physics block types itself; nothing validates it against a math vocabulary
    props = entity_properties(
        models.Entity(type='law', members=[0], id=0), 'book.pdf'
    )
    assert props['type'] == 'law'


def test_entity_properties_omit_unset_attributes():
    props = entity_properties(
        models.Entity(members=[3], id=2),  # untyped, as the block finder emits
        'book.pdf',
    )
    for absent in (
        'type',
        'label',
        'number',
        'title',
        'instruction',
        'contents',
        'bodylist',
    ):
        assert absent not in props


def test_entity_properties_keep_id_zero():
    props = entity_properties(
        models.Entity(type='definition', members=[0], id=0), 'book.pdf'
    )
    assert props['uuid'] == entity_uuid(
        'book.pdf', 0
    )  # a falsy-but-valid id is not dropped


def test_contents_is_native_and_statement_bodylist_is_a_json_string():
    entity = models.Entity(
        type='theorem',
        members=[1],
        id=1,
        contents=['Let n be prime.'],
        bodylist=[
            models.BodySegment(description='Let n be prime.', action='premise')
        ],
    )
    props = entity_properties(entity, 'book.pdf')
    assert props['contents'] == ['Let n be prime.']  # native string array
    assert json.loads(
        props['bodylist']
    ) == [  # statement structure stays on the entity
        {'description': 'Let n be prime.', 'action': 'premise'}
    ]


def test_derivations_and_concepts_are_not_entity_properties():
    # procedures reify into :Procedure/:Event and concepts into :Concept, so neither is a property
    entity = models.Entity(
        type='theorem',
        members=[1],
        id=1,
        procedures=[models.Procedure(type='proof', contents=['Clear.'])],
        concepts=['group theory'],
    )
    props = entity_properties(entity, 'book.pdf')
    assert 'procedures' not in props
    assert 'concepts' not in props


# --- writer planning ---


def test_entity_rows_are_one_flat_batch_carrying_the_type_as_a_property():
    rows = entity_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 3  # no grouping by label — the type is a property
    assert [row['type'] for row in rows] == ['definition', 'theorem', 'problem']
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

    queries = [q for q, _ in calls]
    # one batched MERGE applying the :Mention role label — and NO per-type label
    assert any('SET e:Mention' in q for q in queries)
    assert not any('SET e:Theorem' in q for q in queries)
    # entities are rooted under their :Source via :HAS_ENTITY
    root = next(c for c in calls if ':HAS_ENTITY' in c[0])
    assert root[1]['src'] == source_uuid('book.pdf')
    assert set(root[1]['uuids']) == {
        entity_uuid('book.pdf', i) for i in range(3)
    }
    # members are linked via :DERIVED_FROM, one edge per (entity, member)
    members = next(c for c in calls if ':DERIVED_FROM' in c[0])
    assert len(members[1]['pairs']) == 4


def test_persist_entities_is_a_noop_for_an_empty_overlay(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_entities([], 'book.pdf'))
    assert calls == []  # nothing opened, nothing written
