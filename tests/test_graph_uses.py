"""Step-level :USES mapping — pure, no database (neo4j is stubbed in conftest). Verifies the
name-match locates a reference in the right proof step, rolls up to the correct :Event and
:Canonical uuids, skips fieldless/proofless entities, and that persist_uses issues the edge query."""

import asyncio

from kms.core.models import BodySegment, Entity, EntityType, Proof, Reference
from kms.graph.procedures import event_uuid
from kms.graph.references import canonical_uuid
from kms.graph.uses import _mentions, uses_rows
from kms.graph.writer import persist_uses

# A theorem whose proof cites the Mean Value Theorem in its second step (not its first).
_THEOREM = Entity(
    type=EntityType.THEOREM,
    members=[1],
    id=1,
    refs=[Reference(target="Mean Value Theorem", kind="theorem", tactic="lemma")],
    proofs=[
        Proof(
            bodylist=[
                BodySegment(description="Let f be differentiable.", action="premise"),
                BodySegment(description="By the Mean Value Theorem, f'(c)=0.", action="deduction"),
            ]
        )
    ],
)
_OVERLAY = [_THEOREM]


# --- the name match ---


def test_mentions_matches_whole_token_case_insensitively():
    assert _mentions("By the Mean Value Theorem, done.", "mean value theorem")
    assert _mentions("Take a Set S.", "Set")


def test_mentions_does_not_match_substrings():
    assert not _mentions("Consider the subset T.", "Set")
    assert not _mentions("", "Set")


# --- planning ---


def test_uses_rows_locate_the_reference_in_its_proof_step():
    rows = uses_rows(_OVERLAY, "book.pdf")
    assert len(rows) == 1  # matched only the second step, once
    assert rows[0] == {
        # step index 1 = the second proof step
        "event": event_uuid("book.pdf", 1, "proof", 0, 1),
        "canonical": canonical_uuid("theorem", "Mean Value Theorem"),
        "tactic": "lemma",
    }


def test_uses_rows_skip_entities_without_refs_or_proofs():
    # a problem with a ref but no proof steps contributes no :USES (its ref stays entity-level)
    problem = Entity(
        type=EntityType.PROBLEM,
        members=[2],
        id=2,
        refs=[Reference(target="Set", kind="definition", tactic="premise")],
    )
    assert uses_rows([problem], "book.pdf") == []


def test_uses_rows_dedupe_by_event_and_canonical():
    # the same target named twice in one step yields a single edge
    entity = Entity(
        type=EntityType.THEOREM,
        members=[1],
        id=0,
        refs=[Reference(target="Set", kind="definition", tactic="premise")],
        proofs=[Proof(bodylist=[BodySegment(description="A Set is a Set.", action="premise")])],
    )
    assert len(uses_rows([entity], "book.pdf")) == 1


# --- persist_uses orchestration, via a fake session (no server) ---


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


def test_persist_uses_writes_event_to_canonical_edges(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_uses(_OVERLAY, "book.pdf"))

    queries = [q for q, _ in calls]
    edge = next(c for c in calls if ":USES" in c[0])
    assert "(v:Event" in edge[0] and "(c:Canonical" in edge[0]
    assert len(edge[1]["rows"]) == 1
    assert any("SET u.tactic" in q for q in queries)


def test_persist_uses_is_a_noop_without_matches(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_uses([Entity(type=EntityType.DEFINITION, members=[0], id=0)], "b"))
    assert calls == []  # no refs/proofs -> nothing opened, nothing written
