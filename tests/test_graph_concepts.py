"""Concept-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies concept identity is deterministic and GLOBAL (born canonical), that an entity's
field maps onto an :INSTANCE_OF edge, and that persist_concepts mints :Concept nodes + edges via a
fake session."""

import asyncio

from kms.core.models import Entity, EntityType
from kms.graph.concepts import (
    concept_batches,
    concept_uuid,
    instance_rows,
    normalize_concept,
)
from kms.graph.entities import entity_uuid
from kms.graph.writer import persist_concepts

_OVERLAY = [
    Entity(type=EntityType.DEFINITION, members=[0], id=0, field="algebra"),
    Entity(
        type=EntityType.THEOREM, members=[1], id=1, field="algebra"
    ),  # same field -> one concept
    Entity(type=EntityType.PROBLEM, members=[2], id=2, field="analysis"),
    Entity(type=EntityType.DEFINITION, members=[3], id=3),  # no field -> no concept
]


# --- identity ---


def test_concept_uuid_is_deterministic_and_global_across_sources():
    # A concept is born canonical: same (type, name) -> same uuid regardless of book.
    assert concept_uuid("field", "algebra") == concept_uuid("field", "algebra")


def test_concept_uuid_clusters_case_and_whitespace_variants():
    assert concept_uuid("field", "Algebra") == concept_uuid("field", "  algebra ")


def test_concept_uuid_separates_type_and_distinct_names():
    assert concept_uuid("field", "algebra") != concept_uuid("field", "analysis")


def test_normalize_concept_lowercases_and_collapses_whitespace():
    assert normalize_concept("  Applied   Mathematics ") == "applied mathematics"


# --- planning ---


def test_concept_batches_dedupe_by_uuid_and_group_by_type_label():
    batches = concept_batches(_OVERLAY)
    assert set(batches) == {"Field"}
    # algebra appears twice but collapses to one concept; plus analysis => 2 field concepts.
    assert len(batches["Field"]) == 2
    assert {c["uuid"] for c in batches["Field"]} == {
        concept_uuid("field", "algebra"),
        concept_uuid("field", "analysis"),
    }
    assert batches["Field"][0]["type"] == "field"


def test_instance_rows_are_one_per_entity_concept_skipping_fieldless():
    rows = instance_rows(_OVERLAY, "book.pdf")
    assert len(rows) == 3  # the fieldless definition contributes none
    assert rows[0] == {
        "entity": entity_uuid("book.pdf", 0),
        "concept": concept_uuid("field", "algebra"),
    }
    # both algebra entities point at the SAME concept node
    assert rows[1]["concept"] == rows[0]["concept"]


# --- persist_concepts orchestration, via a fake session (no server) ---


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


def test_persist_concepts_mints_concepts_and_instance_edges(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_concepts(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    # concepts MERGE as :Concept carrying their per-type label
    assert any("SET c:Field" in q for q in queries)
    # instance edges, one per entity concept
    edge_call = next(c for c in calls if ":INSTANCE_OF" in c[0])
    assert len(edge_call[1]["rows"]) == 3


def test_persist_concepts_is_a_noop_without_fields(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_concepts([Entity(type=EntityType.DEFINITION, members=[0], id=0)], "b"))
    assert calls == []  # no fields -> nothing opened, nothing written
