"""Reference-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies hub identity is deterministic and GLOBAL (not source-scoped), that references map
to the right edge rows, and that persist_references mints hubs + edges via a fake session."""

import asyncio

from kms.core.models import Entity, EntityType, Reference
from kms.graph.entities import entity_uuid
from kms.graph.references import hub_batch, hub_uuid, normalize_target, reference_rows
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


# --- hub identity ---


def test_hub_uuid_is_deterministic_and_global_across_sources():
    # The hub is NOT source-scoped: same (kind, target) -> same uuid regardless of book.
    assert hub_uuid("definition", "Set") == hub_uuid("definition", "Set")


def test_hub_uuid_clusters_case_and_whitespace_variants():
    assert hub_uuid("definition", "Set") == hub_uuid("definition", "  set ")
    assert hub_uuid("definition", "Vector Space") == hub_uuid("definition", "vector  space")


def test_hub_uuid_separates_kind_and_distinct_names():
    assert hub_uuid("definition", "Set") != hub_uuid("theorem", "Set")
    assert hub_uuid("definition", "Set") != hub_uuid("definition", "Group")


def test_normalize_target_lowercases_and_collapses_whitespace():
    assert normalize_target("Definition", "  Positive   Definite  Matrix ") == (
        "definition#positive definite matrix"
    )


# --- planning ---


def test_hub_batch_is_deduplicated_by_uuid():
    hubs = hub_batch(_OVERLAY)
    # "Set" and "set" collapse to one hub; plus the Mean Value Theorem hub => 2 total.
    assert len(hubs) == 2
    assert {h["uuid"] for h in hubs} == {
        hub_uuid("definition", "Set"),
        hub_uuid("theorem", "Mean Value Theorem"),
    }


def test_reference_rows_are_one_per_reference_and_carry_the_tactic():
    rows = reference_rows(_OVERLAY, "book.pdf")
    assert len(rows) == 3  # 2 refs on the theorem + 1 on the problem
    assert rows[0] == {
        "entity": entity_uuid("book.pdf", 0),
        "hub": hub_uuid("definition", "Set"),
        "tactic": "premise",
    }
    # the problem's "set" resolves to the SAME hub as the theorem's "Set"
    assert rows[2]["hub"] == hub_uuid("definition", "Set")


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


def test_persist_references_mints_hubs_and_edges(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_references(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    # hubs are MERGEd on the GeneralEntity label
    hub_call = next(c for c in calls if "GeneralEntity" in c[0] and "MERGE" in c[0])
    assert len(hub_call[1]["hubs"]) == 2  # deduplicated
    # edges carry the tactic on the relationship
    edge_call = next(c for c in calls if ":REFERENCES" in c[0])
    assert len(edge_call[1]["rows"]) == 3
    assert any("SET ref.tactic" in q for q in queries)


def test_persist_references_is_a_noop_without_references(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_references([Entity(type=EntityType.DEFINITION, members=[0], id=0)], "b"))
    assert calls == []  # no refs -> nothing opened, nothing written
