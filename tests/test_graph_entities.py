"""Entity-overlay graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies identity is stable/deterministic, that a core.Entity maps onto the expected
Neo4j property shape, and that persist_entities issues the right queries/params via a fake session."""

import asyncio
import json

from kms.core.models import BodySegment, Entity, EntityType, Proof, Solution
from kms.graph.entities import entity_label, entity_properties, entity_uuid
from kms.graph.nodes import node_uuid, source_uuid
from kms.graph.writer import entity_batches, member_pairs, persist_entities

_OVERLAY = [
    Entity(type=EntityType.DEFINITION, members=[0], id=0, title="Group", field="algebra"),
    Entity(type=EntityType.THEOREM, members=[1, 2], id=1, number="2.1"),
    Entity(type=EntityType.PROBLEM, members=[3], id=2, instruction="compute"),
]


# --- identity ---


def test_entity_uuid_is_deterministic_and_distinct_by_id_and_source():
    assert entity_uuid("book.pdf", 3) == entity_uuid("book.pdf", 3)
    assert entity_uuid("book.pdf", 3) != entity_uuid("book.pdf", 4)
    assert entity_uuid("book.pdf", 3) != entity_uuid("other.pdf", 3)


def test_entity_uuid_is_disjoint_from_node_uuid_for_the_same_index():
    # entity#3 and node#3 must not collide — they share a uuid namespace but different keys.
    assert entity_uuid("book.pdf", 3) != node_uuid("book.pdf", 3)


# --- labels ---


def test_entity_label_is_the_capitalized_type():
    assert entity_label(Entity(type=EntityType.THEOREM)) == "Theorem"
    assert entity_label(Entity(type=EntityType.PROBLEM)) == "Problem"
    assert entity_label(Entity(type=EntityType.DEFINITION)) == "Definition"


# --- properties ---


def test_entity_properties_map_identity_source_and_scalars():
    props = entity_properties(_OVERLAY[0], "book.pdf")
    assert props["uuid"] == entity_uuid("book.pdf", 0)
    assert props["source"] == source_uuid("book.pdf")
    assert props["type"] == "definition"
    assert props["title"] == "Group" and props["field"] == "algebra"


def test_entity_properties_omit_unset_attributes():
    props = entity_properties(Entity(type=EntityType.PROBLEM, members=[3], id=2), "book.pdf")
    for absent in ("label", "number", "title", "field", "instruction", "contents", "bodylist"):
        assert absent not in props


def test_entity_properties_keep_id_zero():
    props = entity_properties(Entity(type=EntityType.DEFINITION, members=[0], id=0), "book.pdf")
    assert props["uuid"] == entity_uuid("book.pdf", 0)  # a falsy-but-valid id is not dropped


def test_contents_is_a_native_array_but_nested_attributes_are_json_strings():
    entity = Entity(
        type=EntityType.THEOREM,
        members=[1],
        id=1,
        contents=["Let n be prime."],
        bodylist=[BodySegment(description="Let n be prime.", action="premise")],
        proofs=[
            Proof(
                contents=["Clear."],
                bodylist=[BodySegment(description="Clear.", action="conclusion")],
            )
        ],
    )
    props = entity_properties(entity, "book.pdf")
    assert props["contents"] == ["Let n be prime."]  # native string array
    assert json.loads(props["bodylist"]) == [
        {"description": "Let n be prime.", "action": "premise"}
    ]
    assert json.loads(props["proofs"])[0]["bodylist"][0]["action"] == "conclusion"


def test_solution_nested_attribute_is_a_json_string():
    entity = Entity(
        type=EntityType.PROBLEM, members=[3], id=2, solutions=[Solution(contents=["x = 2"])]
    )
    props = entity_properties(entity, "book.pdf")
    assert json.loads(props["solutions"]) == [{"contents": ["x = 2"]}]


# --- writer planning ---


def test_entity_batches_group_by_per_type_label():
    batches = entity_batches(_OVERLAY, "book.pdf")
    assert set(batches) == {"Definition", "Theorem", "Problem"}
    assert batches["Theorem"][0]["number"] == "2.1"


def test_member_pairs_are_one_per_entity_member():
    pairs = member_pairs(_OVERLAY, "book.pdf")
    assert len(pairs) == 4  # 1 + 2 + 1 members across the overlay
    assert pairs[0] == {"entity": entity_uuid("book.pdf", 0), "node": node_uuid("book.pdf", 0)}
    assert pairs[-1] == {"entity": entity_uuid("book.pdf", 2), "node": node_uuid("book.pdf", 3)}


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
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_entities(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    # each entity MERGE applies its per-type label
    assert any("SET e:Theorem" in q for q in queries)
    # entities are rooted under their :Source via :HAS_ENTITY
    root = next(c for c in calls if ":HAS_ENTITY" in c[0])
    assert root[1]["src"] == source_uuid("book.pdf")
    assert set(root[1]["uuids"]) == {entity_uuid("book.pdf", i) for i in range(3)}
    # members are linked via :DERIVED_FROM, one edge per (entity, member)
    members = next(c for c in calls if ":DERIVED_FROM" in c[0])
    assert len(members[1]["pairs"]) == 4


def test_persist_entities_is_a_noop_for_an_empty_overlay(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_entities([], "book.pdf"))
    assert calls == []  # nothing opened, nothing written
