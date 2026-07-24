"""Procedural-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies procedure/event identity is stable, deterministic and disjoint from the other
uuid namespaces, that proofs/solutions map onto the expected :Procedure/:Event rows, and that
persist_procedures issues the right queries/params via a fake session."""

import asyncio

from kms.core.models import BodySegment, Entity, EntityType, Proof, Solution
from kms.graph.entities import entity_uuid
from kms.graph.nodes import node_uuid
from kms.graph.procedures import (
    event_rows,
    event_uuid,
    first_pairs,
    has_procedure_pairs,
    procedure_batches,
    procedure_label,
    procedure_uuid,
    then_pairs,
)
from kms.graph.writer import persist_procedures

# A theorem with a two-step proof, and a problem with a (stepless) solution.
_THEOREM = Entity(
    type=EntityType.THEOREM,
    members=[1],
    id=1,
    proofs=[
        Proof(
            contents=["Assume n≥3.", "Then Z(Sn) is trivial."],
            bodylist=[
                BodySegment(description="Assume n≥3.", action="assumption"),
                BodySegment(description="Then Z(Sn) is trivial.", action="conclusion"),
            ],
        )
    ],
)
_PROBLEM = Entity(
    type=EntityType.PROBLEM, members=[3], id=2, solutions=[Solution(contents=["x = 2"])]
)
_OVERLAY = [_THEOREM, _PROBLEM]


# --- identity ---


def test_procedure_uuid_is_deterministic_and_distinct_by_kind_and_index():
    assert procedure_uuid("b", 1, "proof", 0) == procedure_uuid("b", 1, "proof", 0)
    assert procedure_uuid("b", 1, "proof", 0) != procedure_uuid("b", 1, "proof", 1)
    assert procedure_uuid("b", 1, "proof", 0) != procedure_uuid("b", 1, "solution", 0)
    assert procedure_uuid("b", 1, "proof", 0) != procedure_uuid("other", 1, "proof", 0)


def test_event_uuid_is_deterministic_and_ordered_within_a_procedure():
    assert event_uuid("b", 1, "proof", 0, 0) == event_uuid("b", 1, "proof", 0, 0)
    assert event_uuid("b", 1, "proof", 0, 0) != event_uuid("b", 1, "proof", 0, 1)


def test_uuids_are_disjoint_from_node_and_entity_namespaces():
    # a procedure/event and a node/entity with matching numeric keys must not collide
    assert procedure_uuid("b", 1, "proof", 0) != entity_uuid("b", 1)
    assert procedure_uuid("b", 1, "proof", 0) != node_uuid("b", 1)
    assert event_uuid("b", 1, "proof", 0, 0) != procedure_uuid("b", 1, "proof", 0)


# --- labels ---


def test_procedure_label_is_the_capitalized_kind():
    assert procedure_label("proof") == "Proof"
    assert procedure_label("solution") == "Solution"


# --- planning ---


def test_procedure_batches_group_by_per_kind_label_with_contents():
    batches = procedure_batches(_OVERLAY, "book.pdf")
    assert set(batches) == {"Proof", "Solution"}
    assert batches["Proof"][0]["uuid"] == procedure_uuid("book.pdf", 1, "proof", 0)
    assert batches["Proof"][0]["contents"] == ["Assume n≥3.", "Then Z(Sn) is trivial."]
    assert batches["Solution"][0]["contents"] == ["x = 2"]


def test_event_rows_are_one_per_proof_step_with_action_and_text():
    rows = event_rows(_OVERLAY, "book.pdf")
    assert len(rows) == 2  # the solution contributes no steps (no bodylist)
    assert rows[0]["action"] == "assumption" and rows[0]["index"] == 0
    assert rows[1]["text"] == "Then Z(Sn) is trivial."
    assert rows[0]["uuid"] == event_uuid("book.pdf", 1, "proof", 0, 0)


def test_has_procedure_pairs_are_one_per_derivation():
    pairs = has_procedure_pairs(_OVERLAY, "book.pdf")
    assert len(pairs) == 2  # one proof + one solution
    assert pairs[0] == {
        "entity": entity_uuid("book.pdf", 1),
        "procedure": procedure_uuid("book.pdf", 1, "proof", 0),
    }


def test_first_pairs_only_for_procedures_with_steps():
    pairs = first_pairs(_OVERLAY, "book.pdf")
    assert len(pairs) == 1  # only the proof has steps; the stepless solution has no :FIRST
    assert pairs[0]["event"] == event_uuid("book.pdf", 1, "proof", 0, 0)


def test_then_pairs_thread_consecutive_steps_and_never_cross_procedures():
    pairs = then_pairs(_OVERLAY, "book.pdf")
    assert len(pairs) == 1  # two steps -> one :THEN edge
    assert pairs[0] == {
        "from": event_uuid("book.pdf", 1, "proof", 0, 0),
        "to": event_uuid("book.pdf", 1, "proof", 0, 1),
    }


# --- persist_procedures orchestration, via a fake session (no server) ---


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


def test_persist_procedures_writes_procedures_events_and_spine(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_procedures(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    assert any("SET p:Proof" in q for q in queries)  # per-kind label applied
    assert any("SET p:Solution" in q for q in queries)
    haspair = next(c for c in calls if ":HAS_PROCEDURE" in c[0])
    assert len(haspair[1]["pairs"]) == 2  # entity -> each derivation
    first = next(c for c in calls if ":FIRST" in c[0])
    assert len(first[1]["pairs"]) == 1  # only the proof has an opening step
    then = next(c for c in calls if ":THEN" in c[0])
    assert len(then[1]["pairs"]) == 1  # one step-to-step edge


def test_persist_procedures_is_a_noop_without_derivations(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_procedures([Entity(type=EntityType.DEFINITION, members=[0], id=0)], "b"))
    assert calls == []  # a definition has no proof/solution, so nothing is opened or written
