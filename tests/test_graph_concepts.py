"""Concept-layer graph mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies concept identity is deterministic and GLOBAL (born canonical), that an entity's
and a procedure step's induced concepts each map onto an :INSTANCE_OF edge, and that persist_concepts
mints :Concept nodes + edges via a fake session."""

import asyncio

from kms.core import models
from kms.graph.concepts import (
    concept_rows,
    concept_uuid,
    entity_instance_rows,
    event_instance_rows,
    normalize_concept,
)
from kms.graph.entities import entity_uuid
from kms.graph.procedures import event_uuid
from kms.graph.writer import persist_concepts

_OVERLAY = [
    models.Entity(
        type='definition', members=[0], id=0, concepts=['group theory']
    ),
    # same concept, different spelling -> one node
    models.Entity(type='theorem', members=[1], id=1, concepts=['Group Theory']),
    models.Entity(type='problem', members=[2], id=2, concepts=['analysis']),
    models.Entity(type='definition', members=[3], id=3),  # untagged -> nothing
]


# --- identity ---


def test_concept_uuid_is_deterministic_and_global_across_sources():
    # A concept is born canonical: same name -> same uuid regardless of book.
    assert concept_uuid('topic', 'algebra') == concept_uuid('topic', 'algebra')


def test_concept_uuid_clusters_case_and_whitespace_variants():
    assert concept_uuid('topic', 'Algebra') == concept_uuid(
        'topic', '  algebra '
    )


def test_concept_uuid_separates_distinct_names():
    assert concept_uuid('topic', 'algebra') != concept_uuid('topic', 'analysis')


def test_normalize_concept_lowercases_and_collapses_whitespace():
    assert (
        normalize_concept('  Applied   Mathematics ') == 'applied mathematics'
    )


# --- planning ---


def test_concept_rows_dedupe_by_uuid_across_entities():
    rows = concept_rows(_OVERLAY, 'book.pdf')
    # "group theory" appears twice (differently cased) but collapses to one concept; plus analysis
    assert {row['uuid'] for row in rows} == {
        concept_uuid('topic', 'group theory'),
        concept_uuid('topic', 'analysis'),
    }
    assert rows[0]['type'] == 'topic'


def test_entity_instance_rows_are_one_per_tag_skipping_untagged():
    rows = entity_instance_rows(_OVERLAY, 'book.pdf')
    assert len(rows) == 3  # the untagged definition contributes none
    assert rows[0] == {
        'entity': entity_uuid('book.pdf', 0),
        'concept': concept_uuid('topic', 'group theory'),
    }
    # both group-theory entities point at the SAME concept node
    assert rows[1]['concept'] == rows[0]['concept']


def test_event_instance_rows_tag_procedure_steps_on_their_event_uuid():
    entity = models.Entity(
        type='theorem',
        members=[0],
        id=0,
        procedures=[
            models.Procedure(
                type='proof',
                steps=[
                    models.BodySegment(
                        description='Assume not.',
                        action='assumption',
                        concepts=['proof by contradiction'],
                    )
                ],
            )
        ],
    )
    rows = event_instance_rows([entity], 'book.pdf')
    assert rows == [
        {
            'event': event_uuid('book.pdf', 0, 'proof', 0, 0),
            'concept': concept_uuid('topic', 'proof by contradiction'),
        }
    ]


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
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_concepts(_OVERLAY, 'book.pdf'))

    queries = [q for q, _ in calls]
    # concepts MERGE as bare :Concept — the type is a property, not a per-type label
    assert any('MERGE (c:Concept' in q for q in queries)
    assert not any('SET c:Topic' in q for q in queries)
    # instance edges, one per entity concept
    edge_call = next(c for c in calls if ':INSTANCE_OF' in c[0])
    assert len(edge_call[1]['rows']) == 3


def test_persist_concepts_is_a_noop_without_tags(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(
        persist_concepts(
            [models.Entity(type='definition', members=[0], id=0)], 'b'
        )
    )
    assert calls == []  # no concepts -> nothing opened, nothing written
