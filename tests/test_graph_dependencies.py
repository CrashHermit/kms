"""Concept :DEPENDS_ON mapping and writer planning — pure, no database (neo4j is stubbed in
conftest). Verifies a dependency keys onto the SAME global concept uuids the concept layer mints, that
self-dependencies and duplicates are dropped, and that persist_dependencies issues the edge query."""

import asyncio

from kms.core import models
from kms.graph.concepts import concept_uuid
from kms.graph.dependencies import dependency_rows
from kms.graph.writer import persist_dependencies

_DEPENDENCIES = [
    models.Dependency(
        dependent='eigenvalue', prerequisite='vector space', support=3
    ),
    models.Dependency(
        dependent='eigenvalue', prerequisite='linear map', support=1
    ),
]


# --- planning ---


def test_rows_key_onto_the_concept_uuids_the_concept_layer_mints():
    rows = dependency_rows(_DEPENDENCIES)
    assert rows[0] == {
        'dependent': concept_uuid('topic', 'eigenvalue'),
        'prerequisite': concept_uuid('topic', 'vector space'),
        'support': 3,
    }


def test_a_self_dependency_is_dropped():
    # both ends normalizing to one concept would be a loop; a prerequisite graph must stay a DAG
    assert (
        dependency_rows(
            [models.Dependency(dependent='Algebra', prerequisite='algebra')]
        )
        == []
    )


def test_duplicate_pairs_collapse_keeping_the_first_seen():
    rows = dependency_rows(
        [
            models.Dependency(
                dependent='eigenvalue', prerequisite='Vector Space', support=3
            ),
            models.Dependency(
                dependent='Eigenvalue', prerequisite='vector space', support=1
            ),
        ]
    )
    assert len(rows) == 1 and rows[0]['support'] == 3


# --- persist_dependencies orchestration, via a fake session (no server) ---


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


def test_persist_dependencies_matches_both_ends_and_never_mints(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_dependencies(_DEPENDENCIES))

    query, params = calls[0]
    assert 'MERGE (a)-[d:DEPENDS_ON]->(b)' in query
    assert 'MERGE (a:Concept' not in query  # both endpoints are MATCHed
    assert len(params['rows']) == 2


def test_persist_dependencies_is_a_noop_without_dependencies(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_dependencies([]))
    assert calls == []  # nothing opened, nothing written
