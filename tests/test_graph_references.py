"""models.Reference-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies canonical identity is deterministic and GLOBAL (not source-scoped), that references
map to the right edge rows, and that persist_references mints :Entity:Canonical targets + edges via a
fake session."""

import asyncio

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.references import (
    canonical_rows,
    canonical_uuid,
    normalize_target,
    reference_rows,
)
from kms.graph.writer import persist_references

_OVERLAY = [
    models.Entity(
        type='theorem',
        members=[1],
        id=0,
        refs=[
            models.Reference(
                target='Set', kind='definition', relation='assumes'
            ),
            models.Reference(
                target='Mean Value Theorem',
                kind='theorem',
                relation='follows from',
            ),
        ],
    ),
    models.Entity(
        type='problem',
        members=[2],
        id=1,
        refs=[
            models.Reference(
                target='set', kind='definition', relation='applies'
            )
        ],  # dup of "Set"
    ),
]


# --- canonical identity ---


def test_canonical_uuid_is_deterministic_and_global_across_sources():
    # The canonical is NOT source-scoped: same (kind, target) -> same uuid regardless of book.
    assert canonical_uuid('definition', 'Set') == canonical_uuid(
        'definition', 'Set'
    )


def test_canonical_uuid_clusters_case_and_whitespace_variants():
    assert canonical_uuid('definition', 'Set') == canonical_uuid(
        'definition', '  set '
    )
    assert canonical_uuid('definition', 'Vector Space') == canonical_uuid(
        'definition', 'vector  space'
    )


def test_canonical_uuid_separates_kind_and_distinct_names():
    assert canonical_uuid('definition', 'Set') != canonical_uuid(
        'theorem', 'Set'
    )
    assert canonical_uuid('definition', 'Set') != canonical_uuid(
        'definition', 'Group'
    )


def test_normalize_target_lowercases_and_collapses_whitespace():
    assert normalize_target('Definition', '  Positive   Definite  Matrix ') == (
        'definition#positive definite matrix'
    )


# --- planning ---


def test_canonical_rows_dedupe_by_uuid_and_carry_the_kind_as_a_property():
    rows = canonical_rows(_OVERLAY)
    # "Set" and "set" collapse to one canonical; the Mean Value Theorem is a second.
    assert {row['uuid'] for row in rows} == {
        canonical_uuid('definition', 'Set'),
        canonical_uuid('theorem', 'Mean Value Theorem'),
    }
    assert rows[0]['type'] == 'definition'  # typed like a mention definition


def test_reference_rows_are_one_per_reference_and_carry_the_open_relation():
    rows = reference_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 3  # 2 refs on the theorem + 1 on the problem
    assert (
        rows[0]
        == {
            'entity': entity_uuid('book.pdf', 0),
            'canonical': canonical_uuid('definition', 'Set'),
            'relation': 'assumes',  # an open, LLM-named label, not one of nine tactics
        }
    )
    # the problem's "set" resolves to the SAME canonical as the theorem's "Set"
    assert rows[2]['canonical'] == canonical_uuid('definition', 'Set')


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
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_references(_OVERLAY, 'book.pdf'))

    queries = [q for q, _ in calls]
    # canonicals are MERGEd as :Entity carrying the :Canonical role label; the kind is a property
    assert any('SET c:Canonical' in q for q in queries)
    assert not any('SET c:Definition' in q for q in queries)
    # edges carry the relation on the relationship, matching by the :Canonical role label
    edge_call = next(c for c in calls if ':REFERENCES' in c[0])
    assert len(edge_call[1]['rows']) == 3
    assert any('SET ref.relation' in q for q in queries)


def test_persist_references_is_a_noop_without_references(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(
        persist_references(
            [models.Entity(type='definition', members=[0], id=0)], 'b'
        )
    )
    assert calls == []  # no refs -> nothing opened, nothing written
