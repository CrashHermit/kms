"""Procedural-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies procedure/act identity is stable, deterministic and disjoint from the other
uuid namespaces, that derivations map onto the expected :Procedure/:Act rows, and that
persist_procedures issues the right queries/params via a fake session."""

import asyncio

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.nodes import node_uuid
from kms.graph.procedures import (
    act_rows,
    act_uuid,
    first_pairs,
    has_procedure_pairs,
    procedure_member_pairs,
    procedure_rows,
    procedure_uuid,
    then_pairs,
)
from kms.graph.writer import persist_procedures

# A theorem with a two-step proof, and an example with a two-step solution — decomposition is
# universal, so the solution has steps too (the old schema left every solution stepless).
_THEOREM = models.Entity(
    type='theorem',
    members=[1],
    id=1,
    procedures=[
        models.Procedure(
            index=0,
            members=[2],
            contents=['Assume n≥3.', 'Then Z(Sn) is trivial.'],
            steps=['Assume n≥3.', 'Then Z(Sn) is trivial.'],
        )
    ],
)
_EXAMPLE = models.Entity(
    type='example',
    members=[3],
    id=2,
    procedures=[
        models.Procedure(
            index=0,
            members=[4],
            contents=['Substitute x = 2.', 'So y = 5.'],
            steps=['Substitute x = 2.', 'So y = 5.'],
        )
    ],
)
_OVERLAY = [_THEOREM, _EXAMPLE]


# --- identity ---


def test_procedure_uuid_is_deterministic_and_distinct_by_index_and_source():
    assert procedure_uuid('b', 1, 0) == procedure_uuid('b', 1, 0)
    assert procedure_uuid('b', 1, 0) != procedure_uuid('b', 1, 1)
    assert procedure_uuid('b', 1, 0) != procedure_uuid('b', 2, 0)
    assert procedure_uuid('b', 1, 0) != procedure_uuid('other', 1, 0)


def test_act_uuid_is_deterministic_and_ordered_within_a_procedure():
    assert act_uuid('b', 1, 0, 0) == act_uuid('b', 1, 0, 0)
    assert act_uuid('b', 1, 0, 0) != act_uuid('b', 1, 0, 1)
    assert act_uuid('b', 1, 0, 0) != act_uuid('b', 1, 1, 0)


def test_uuids_are_disjoint_from_node_and_entity_namespaces():
    # a procedure/act and a node/entity with matching numeric keys must not collide
    assert procedure_uuid('b', 1, 0) != entity_uuid('b', 1)
    assert procedure_uuid('b', 1, 0) != node_uuid('b', 1)
    assert act_uuid('b', 1, 0, 0) != procedure_uuid('b', 1, 0)


# --- planning ---


def test_procedure_rows_are_one_per_derivation_with_contents_and_no_type():
    rows = procedure_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 2
    assert rows[0]['uuid'] == procedure_uuid('book.pdf', 1, 0)
    assert rows[0]['contents'] == ['Assume n≥3.', 'Then Z(Sn) is trivial.']
    # proof/solution is derivable from the owning entity's type, so it is not stored
    assert 'type' not in rows[0]


def test_act_rows_cover_every_procedure_including_solutions():
    rows = act_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 4  # decomposition is universal: 2 proof + 2 solution
    assert rows[0]['text'] == 'Assume n≥3.' and rows[0]['index'] == 0
    assert rows[0]['uuid'] == act_uuid('book.pdf', 1, 0, 0)
    assert rows[3]['text'] == 'So y = 5.'
    assert 'action' not in rows[0]  # the closed tactic taxonomy is gone


def test_has_procedure_pairs_are_one_per_derivation():
    pairs = has_procedure_pairs(_OVERLAY, 'book.pdf')
    assert len(pairs) == 2
    assert pairs[0] == {
        'entity': entity_uuid('book.pdf', 1),
        'procedure': procedure_uuid('book.pdf', 1, 0),
    }


def test_procedures_carry_their_own_derived_from_provenance():
    pairs = procedure_member_pairs(_OVERLAY, 'book.pdf')
    assert len(pairs) == 2  # one member node each
    assert pairs[0] == {
        'procedure': procedure_uuid('book.pdf', 1, 0),
        'node': node_uuid('book.pdf', 2),
    }


def test_first_pairs_only_for_procedures_with_steps():
    pairs = first_pairs(_OVERLAY, 'book.pdf')
    assert len(pairs) == 2
    assert pairs[0]['act'] == act_uuid('book.pdf', 1, 0, 0)
    stepless = [
        models.Entity(
            type='theorem', members=[1], id=1, procedures=[models.Procedure()]
        )
    ]
    assert first_pairs(stepless, 'b') == []


def test_then_pairs_thread_consecutive_steps_and_never_cross_procedures():
    pairs = then_pairs(_OVERLAY, 'book.pdf')
    assert (
        len(pairs) == 2
    )  # two steps in each of two procedures -> one edge each
    assert pairs[0] == {
        'from': act_uuid('book.pdf', 1, 0, 0),
        'to': act_uuid('book.pdf', 1, 0, 1),
    }
    assert pairs[1]['from'] == act_uuid('book.pdf', 2, 0, 0)


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


def test_persist_procedures_writes_procedures_acts_and_spine(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_procedures(_OVERLAY, 'book.pdf'))

    queries = [query for query, _ in calls]
    assert any('MERGE (p:Procedure' in query for query in queries)
    assert any('MERGE (a:Act' in query for query in queries)
    # no per-kind label is applied — :Procedure and :Act are bare
    assert not any('SET p:Proof' in query for query in queries)
    owners = next(call for call in calls if ':HAS_PROCEDURE' in call[0])
    assert len(owners[1]['pairs']) == 2
    derived = next(
        call
        for call in calls
        if ':DERIVED_FROM' in call[0] and 'p:Procedure' in call[0]
    )
    assert len(derived[1]['pairs']) == 2
    first = next(call for call in calls if ':FIRST' in call[0])
    assert len(first[1]['pairs']) == 2
    then = next(call for call in calls if ':THEN' in call[0])
    assert len(then[1]['pairs']) == 2


def test_persist_procedures_is_a_noop_without_derivations(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(
        persist_procedures(
            [models.Entity(type='definition', members=[0], id=0)], 'b'
        )
    )
    assert calls == []  # a definition has no derivation — nothing is opened
