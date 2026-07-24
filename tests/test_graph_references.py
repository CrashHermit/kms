"""Reference-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies canonical identity is deterministic and GLOBAL (not source-scoped), that references
map to the right edge rows, and that persist_references mints :Entity:Canonical targets + edges via a
fake session."""

import asyncio

from kms.core.models import Entity, EntityType, Reference
from kms.graph.entities import entity_uuid
from kms.graph.references import (
    canonical_batches,
    canonical_uuid,
    normalize_target,
    reference_rows,
)
from kms.graph.writer import persist_references

_OVERLAY = [
    Entity(
        type=EntityType.THEOREM,
        members=[1],
        id=0,
        refs=[
            Reference(target="Set", kind="definition", tactic="premise"),
            Reference(target="Mean Value Theorem", kind="theorem", tactic="lemma"),
        ],
    ),
    Entity(
        type=EntityType.PROBLEM,
        members=[2],
        id=1,
        refs=[Reference(target="set", kind="definition", tactic="deduction")],  # dup of "Set"
    ),
]


# --- canonical identity ---


def test_canonical_uuid_is_deterministic_and_global_across_sources():
    # The canonical is NOT source-scoped: same (kind, target) -> same uuid regardless of book.
    assert canonical_uuid("definition", "Set") == canonical_uuid("definition", "Set")


def test_canonical_uuid_clusters_case_and_whitespace_variants():
    assert canonical_uuid("definition", "Set") == canonical_uuid("definition", "  set ")
    assert canonical_uuid("definition", "Vector Space") == canonical_uuid(
        "definition", "vector  space"
    )


def test_canonical_uuid_separates_kind_and_distinct_names():
    assert canonical_uuid("definition", "Set") != canonical_uuid("theorem", "Set")
    assert canonical_uuid("definition", "Set") != canonical_uuid("definition", "Group")


def test_normalize_target_lowercases_and_collapses_whitespace():
    assert normalize_target("Definition", "  Positive   Definite  Matrix ") == (
        "definition#positive definite matrix"
    )


# --- planning ---


def test_canonical_batches_dedupe_by_uuid_and_group_by_type_label():
    batches = canonical_batches(_OVERLAY)
    # "Set" and "set" collapse to one Definition canonical; the Mean Value Theorem is a Theorem one.
    assert set(batches) == {"Definition", "Theorem"}
    assert len(batches["Definition"]) == 1
    assert batches["Definition"][0]["uuid"] == canonical_uuid("definition", "Set")
    assert batches["Definition"][0]["type"] == "definition"  # typed like a mention definition
    assert batches["Theorem"][0]["uuid"] == canonical_uuid("theorem", "Mean Value Theorem")


def test_reference_rows_are_one_per_reference_and_carry_the_tactic():
    rows = reference_rows(_OVERLAY, "book.pdf")
    assert len(rows) == 3  # 2 refs on the theorem + 1 on the problem
    assert rows[0] == {
        "entity": entity_uuid("book.pdf", 0),
        "canonical": canonical_uuid("definition", "Set"),
        "tactic": "premise",
    }
    # the problem's "set" resolves to the SAME canonical as the theorem's "Set"
    assert rows[2]["canonical"] == canonical_uuid("definition", "Set")


# --- persist_references orchestration, via a fake session (no server) ---


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


def test_persist_references_mints_canonicals_and_edges(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_references(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    # canonicals are MERGEd as :Entity carrying the :Canonical role label + their per-type label
    assert any("SET c:Canonical SET c:Definition" in q for q in queries)
    assert any("SET c:Canonical SET c:Theorem" in q for q in queries)
    # edges carry the tactic on the relationship, matching the target by its :Canonical role label
    edge_call = next(c for c in calls if ":REFERENCES" in c[0])
    assert len(edge_call[1]["rows"]) == 3
    assert any("SET ref.tactic" in q for q in queries)


def test_persist_references_is_a_noop_without_references(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_references([Entity(type=EntityType.DEFINITION, members=[0], id=0)], "b"))
    assert calls == []  # no refs -> nothing opened, nothing written
