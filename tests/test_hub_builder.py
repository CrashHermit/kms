import asyncio
from types import SimpleNamespace

import pytest

from kms.construction import hub_builder
from kms.graph import hubs


@pytest.fixture(autouse=True)
def _stub_lexical_rebuild(monkeypatch):
    async def fake_rebuild(*args, **kwargs):
        return {'name_hubs': 0, 'names': 0}

    monkeypatch.setattr(hub_builder.name_hubs, 'rebuild', fake_rebuild)
    monkeypatch.setattr(
        hub_builder.name_hubs,
        'rebuild_meta',
        lambda *args, **kwargs: asyncio.sleep(
            0, result={'name_hubs': 0, 'source_name_hubs': 0}
        ),
    )


def test_build_hubs_empty_result():
    result = asyncio.run(
        hub_builder.build_hubs(
            'entity',
            [],
            spec=hub_builder.SOURCE_SPEC,
            adjudicator=object(),
            synthesizer=object(),
        )
    )

    assert result == {
        'clusters': 0,
        'records': 0,
        'hubs': [],
        'subsumption_edges': [],
    }


def test_meta_hub_ids_are_membership_stable():
    records = [
        {'uuid': 'source-hub-b'},
        {'uuid': 'source-hub-a'},
    ]

    assert hub_builder._meta_hub_id('entity', records, {}) == (
        hub_builder._meta_hub_id('entity', list(reversed(records)), {})
    )


def test_rebuild_replaces_only_the_requested_source(monkeypatch):
    calls = []

    async def fake_all_components(session_factory, kind, source=None):
        calls.append(('read', kind, source))
        return [
            {
                'uuid': 'entity-a',
                'name': 'graph',
                'node_id': 1,
                'source': source,
                'embedding': [1.0],
            }
        ]

    async def fake_build_hubs(kind, records, **kwargs):
        calls.append(('build_hubs', kind, kwargs['spec'].tier))
        return {
            'clusters': 1,
            'records': len(records),
            'hubs': [{'uuid': 'hub-a'}],
            'subsumption_edges': [],
        }

    async def fake_clear(kind, source, *, session_factory):
        calls.append(('clear', kind, source))

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', kind, hubs, kwargs['tier']))

    monkeypatch.setattr(
        hub_builder.queries, 'all_components', fake_all_components
    )
    monkeypatch.setattr(hub_builder, 'build_hubs', fake_build_hubs)
    monkeypatch.setattr(hub_builder.writer, 'clear_source_hubs', fake_clear)
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(
        hub_builder.triplet_hubs,
        'rebuild',
        lambda **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        hub_builder.name_hubs,
        'rebuild',
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    result = asyncio.run(
        hub_builder.rebuild_hubs(
            'entity',
            language_model=object(),
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
            source='book-a',
        )
    )

    assert result == {'clusters': 1, 'records': 1}
    assert calls == [
        ('read', 'entity', 'book-a'),
        ('build_hubs', 'entity', 'source'),
        ('clear', 'entity', 'book-a'),
        ('persist', 'entity', [{'uuid': 'hub-a'}], 'source'),
    ]


def test_rebuild_meta_replaces_the_disposable_meta_layer(monkeypatch):
    calls = []

    async def fake_source_hubs(session_factory, kind):
        calls.append(('read', kind))
        return [
            {
                'uuid': 'source-hub-a',
                'name': 'graph',
                'aliases': ['graph'],
                'description': 'A graph.',
                'embedding': [1.0],
                'source': 'book-a',
            },
            {
                'uuid': 'source-hub-b',
                'name': 'graph',
                'aliases': ['graph'],
                'description': 'A graph.',
                'embedding': [1.0],
                'source': 'book-b',
            },
        ]

    async def fake_build_hubs(kind, records, **kwargs):
        calls.append(('build_hubs', kind, kwargs['spec']))
        return {
            'clusters': 1,
            'hubs': [
                {
                    'uuid': 'meta-hub-a',
                    'members': ['source-hub-a', 'source-hub-b'],
                }
            ],
            'subsumption_edges': [],
        }

    async def fake_clear(kind, *, session_factory):
        calls.append(('clear', kind))

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', kind, hubs, kwargs['tier']))

    monkeypatch.setattr(
        hub_builder.queries, 'all_source_hubs', fake_source_hubs
    )
    monkeypatch.setattr(hub_builder, 'build_hubs', fake_build_hubs)
    monkeypatch.setattr(hub_builder.writer, 'clear_meta_hubs', fake_clear)
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(
        hub_builder.triplet_hubs,
        'rebuild_meta',
        lambda **kwargs: asyncio.sleep(0),
    )

    result = asyncio.run(
        hub_builder.rebuild_meta_hubs(
            'entity',
            language_model=object(),
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
        )
    )

    assert result == {'clusters': 1, 'hubs': 1}
    assert calls[0] == ('read', 'entity')
    assert calls[1] == (
        'build_hubs',
        'entity',
        hub_builder.META_SPEC,
    )
    assert calls[2:] == [
        ('clear', 'entity'),
        (
            'persist',
            'entity',
            [
                {
                    'uuid': 'meta-hub-a',
                    'members': ['source-hub-a', 'source-hub-b'],
                }
            ],
            'meta',
        ),
    ]


def test_rebuild_meta_requires_two_distinct_sources(monkeypatch):
    async def fake_source_hubs(session_factory, kind):
        return [
            {
                'uuid': 'source-hub-a',
                'name': 'graph',
                'embedding': [1.0],
                'source': 'book-a',
            },
            {
                'uuid': 'source-hub-b',
                'name': 'vertex',
                'embedding': [1.0],
                'source': 'book-a',
            },
        ]

    monkeypatch.setattr(
        hub_builder.queries, 'all_source_hubs', fake_source_hubs
    )

    with pytest.raises(
        RuntimeError, match='requires at least two distinct sources'
    ):
        asyncio.run(
            hub_builder.rebuild_meta_hubs(
                'entity',
                language_model=object(),
                adjudicator=object(),
                synthesizer=object(),
                session_factory=object(),
            )
        )


def test_meta_qualification_discards_same_source_clusters():
    result = hub_builder._qualify_meta_result(
        {
            'clusters': 2,
            'records': 3,
            'hubs': [
                {'uuid': 'same-source', 'members': ['hub-a', 'hub-b']},
                {'uuid': 'cross-source', 'members': ['hub-a', 'hub-c']},
            ],
            'subsumption_edges': [
                {'general': 'same-source', 'specific': 'cross-source'}
            ],
        },
        [
            {'uuid': 'hub-a', 'source': 'book-a'},
            {'uuid': 'hub-b', 'source': 'book-a'},
            {'uuid': 'hub-c', 'source': 'book-b'},
        ],
    )

    assert result['hubs'] == [
        {'uuid': 'cross-source', 'members': ['hub-a', 'hub-c']}
    ]
    assert result['subsumption_edges'] == []


def test_engine_builds_source_and_meta_records(monkeypatch):
    records = [
        {
            'uuid': 'source-entity-a',
            'name': 'graph',
            'aliases': ['$G'],
            'description': 'A mathematical graph.',
            'source': 'book-a',
            'embedding': [1.0, 0.0],
        }
    ]

    monkeypatch.setattr(
        hub_builder,
        '_coarse_clusters',
        lambda values, threshold: [values],
    )

    async def fake_adjudicate(*args):
        return [args[0]], []

    async def fake_definitions(*args):
        return [{'canonical_name': 'graph', 'description': 'A graph.'}]

    class _Embedder:
        async def embed(self, values):
            return [[0.5, 0.5] for _ in values]

    monkeypatch.setattr(hub_builder, '_adjudicate_component', fake_adjudicate)
    monkeypatch.setattr(
        hub_builder, '_synthesize_definitions', fake_definitions
    )
    monkeypatch.setattr(
        hub_builder.llm,
        'gate',
        lambda max_concurrency: asyncio.Semaphore(1),
    )
    monkeypatch.setattr(
        hub_builder.embeddings,
        'embedder',
        lambda: _Embedder(),
    )

    source_result = asyncio.run(
        hub_builder.build_hubs(
            'entity',
            records,
            spec=hub_builder.SOURCE_SPEC,
            adjudicator=object(),
            synthesizer=object(),
        )
    )
    meta_result = asyncio.run(
        hub_builder.build_hubs(
            'entity',
            [
                {
                    **records[0],
                    'name': 'graph',
                    'source': None,
                }
            ],
            spec=hub_builder.META_SPEC,
            adjudicator=object(),
            synthesizer=object(),
        )
    )

    source_hub = source_result['hubs'][0]
    assert source_hub['uuid'] == hubs.hub_uuid(
        'entity', 'book-a', 'source-entity-a'
    )
    assert source_hub['source'] == 'book-a'
    assert source_hub['aliases'] == ['$G', 'graph']
    assert meta_result['hubs'][0]['uuid'] == hubs.meta_hub_uuid(
        'entity', 'source-entity-a'
    )
    assert meta_result['hubs'][0]['source'] is None


def test_assign_source_creates_a_new_hub_when_candidates_are_separate(
    monkeypatch,
):
    calls = []

    async def fake_unassigned(session_factory, kind, source):
        return [
            {
                'uuid': 'entity-a',
                'name': 'vertex',
                'aliases': [],
                'description': 'A vertex.',
                'embedding': [0.7, 0.7],
                'source': source,
            }
        ]

    async def fake_search(*args, **kwargs):
        return [
            {
                'uuid': 'hub-a',
                'canonical_name': 'graph',
                'aliases': ['graph'],
                'description': 'A graph.',
                'embedding': [0.7, 0.7],
                'score': 0.5,
            }
        ]

    class _Judge:
        async def aforward(self, **kwargs):
            return SimpleNamespace(decision='Separate', more_general='none')

    async def fake_new_hub(kind, record, synthesizer, gate, spec):
        return {
            'uuid': 'hub-b',
            'source': 'book-a',
            'canonical_name': 'vertex',
            'aliases': ['vertex'],
            'description': 'A vertex.',
            'embedding': [0.7, 0.7],
            'members': ['entity-a'],
        }

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', hubs, kwargs['subsumption_edges']))

    async def fake_attach(kind, assignments, **kwargs):
        calls.append(('attach', assignments))

    monkeypatch.setattr(
        hub_builder.queries, 'unassigned_components', fake_unassigned
    )
    monkeypatch.setattr(hub_builder.queries, 'vector_search', fake_search)
    monkeypatch.setattr(hub_builder, '_new_hub', fake_new_hub)
    monkeypatch.setattr(
        hub_builder,
        '_choose_hub',
        lambda *args, **kwargs: asyncio.sleep(
            0, result=(None, [], 0.0, 'Separate')
        ),
    )
    monkeypatch.setattr(
        hub_builder,
        '_refresh_source_hubs',
        lambda *args, **kwargs: asyncio.sleep(
            0,
            result=[
                {
                    'uuid': 'hub-b',
                    'source': 'book-a',
                    'canonical_name': 'vertex',
                    'aliases': ['vertex'],
                    'description': 'A vertex.',
                    'embedding': [0.7, 0.7],
                    'members': ['entity-a'],
                }
            ],
        ),
    )
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(
        hub_builder.writer, 'attach_source_components', fake_attach
    )
    monkeypatch.setattr(
        hub_builder.llm,
        'gate',
        lambda max_concurrency: asyncio.Semaphore(1),
    )

    result = asyncio.run(
        hub_builder.assign_source_hubs(
            'entity',
            'book-a',
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
        )
    )

    assert result == {
        'assigned': 1,
        'new_hubs': 1,
        'hierarchies': 0,
        'changed_hubs': ['hub-b'],
    }
    assert calls == [
        (
            'persist',
            [
                {
                    'uuid': 'hub-b',
                    'source': 'book-a',
                    'canonical_name': 'vertex',
                    'aliases': ['vertex'],
                    'description': 'A vertex.',
                    'embedding': [0.7, 0.7],
                    'members': ['entity-a'],
                }
            ],
            [],
        ),
        ('attach', [{'component': 'entity-a', 'hub': 'hub-b'}]),
    ]


def test_assign_source_attaches_to_a_high_similarity_hub(monkeypatch):
    calls = []

    async def fake_unassigned(session_factory, kind, source):
        return [
            {
                'uuid': 'entity-a',
                'name': 'graph',
                'aliases': [],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'source': source,
            }
        ]

    async def fake_search(*args, **kwargs):
        return [
            {
                'uuid': 'hub-a',
                'canonical_name': 'graph',
                'aliases': ['graph'],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'score': 0.95,
            }
        ]

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', hubs))

    async def fake_attach(kind, assignments, **kwargs):
        calls.append(('attach', assignments, kwargs['aliases']))

    monkeypatch.setattr(
        hub_builder.queries, 'unassigned_components', fake_unassigned
    )
    monkeypatch.setattr(hub_builder.queries, 'vector_search', fake_search)
    monkeypatch.setattr(
        hub_builder,
        '_refresh_source_hubs',
        lambda *args, **kwargs: asyncio.sleep(0, result=[]),
    )
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(
        hub_builder.writer, 'attach_source_components', fake_attach
    )
    monkeypatch.setattr(
        hub_builder.llm,
        'gate',
        lambda max_concurrency: asyncio.Semaphore(1),
    )

    result = asyncio.run(
        hub_builder.assign_source_hubs(
            'entity',
            'book-a',
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
        )
    )

    assert result == {
        'assigned': 1,
        'new_hubs': 0,
        'hierarchies': 0,
        'changed_hubs': ['hub-a'],
    }
    assert calls == [
        ('persist', []),
        (
            'attach',
            [{'component': 'entity-a', 'hub': 'hub-a'}],
            [{'hub': 'hub-a', 'aliases': ['graph']}],
        ),
    ]


def test_assign_source_considers_hubs_created_earlier_in_the_batch(
    monkeypatch,
):
    calls = []

    async def fake_unassigned(session_factory, kind, source):
        return [
            {
                'uuid': 'entity-a',
                'name': 'graph',
                'aliases': [],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'source': source,
            },
            {
                'uuid': 'entity-b',
                'name': 'Graph',
                'aliases': [],
                'description': 'The same graph.',
                'embedding': [1.0, 0.0],
                'source': source,
            },
        ]

    async def fake_search(*args, **kwargs):
        return []

    async def fake_new_hub(kind, record, synthesizer, gate, spec):
        return {
            'uuid': 'hub-a',
            'source': 'book-a',
            'canonical_name': 'graph',
            'aliases': ['graph'],
            'description': 'A graph.',
            'embedding': [1.0, 0.0],
            'members': [record['uuid']],
        }

    async def fake_refresh(
        kind, records_by_hub, new_hub_ids, synthesizer, gate
    ):
        return [
            {
                'uuid': 'hub-a',
                'source': 'book-a',
                'canonical_name': 'graph',
                'aliases': ['Graph', 'graph'],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'members': ['entity-a', 'entity-b'],
            }
        ]

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', hubs))

    async def fake_attach(kind, assignments, **kwargs):
        calls.append(('attach', assignments))

    monkeypatch.setattr(
        hub_builder.queries, 'unassigned_components', fake_unassigned
    )
    monkeypatch.setattr(hub_builder.queries, 'vector_search', fake_search)
    monkeypatch.setattr(hub_builder, '_new_hub', fake_new_hub)
    monkeypatch.setattr(hub_builder, '_refresh_source_hubs', fake_refresh)
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(
        hub_builder.writer, 'attach_source_components', fake_attach
    )
    monkeypatch.setattr(
        hub_builder.llm,
        'gate',
        lambda max_concurrency: asyncio.Semaphore(1),
    )

    result = asyncio.run(
        hub_builder.assign_source_hubs(
            'entity',
            'book-a',
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
        )
    )

    assert result == {
        'assigned': 2,
        'new_hubs': 1,
        'hierarchies': 0,
        'changed_hubs': ['hub-a'],
    }
    assert calls[1] == (
        'attach',
        [
            {'component': 'entity-a', 'hub': 'hub-a'},
            {'component': 'entity-b', 'hub': 'hub-a'},
        ],
    )


def test_align_meta_persists_only_cross_source_assignments(monkeypatch):
    calls = []

    async def fake_source_hubs(session_factory, kind, hub_uuids=None):
        if hub_uuids is not None:
            assert hub_uuids == ['source-hub-a', 'source-hub-b']
        return [
            {
                'uuid': 'source-hub-a',
                'name': 'graph',
                'aliases': ['$G'],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'source': 'book-a',
            },
            {
                'uuid': 'source-hub-b',
                'name': 'graph',
                'aliases': [],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'source': 'book-b',
            },
        ]

    async def fake_search(*args, **kwargs):
        assert kwargs['index_name'] == 'meta_entity_hub_embedding'
        assert 'source' not in kwargs
        return [
            {
                'uuid': 'meta-hub-a',
                'canonical_name': 'graph',
                'aliases': ['graph'],
                'description': 'A graph.',
                'embedding': [1.0, 0.0],
                'score': 0.95,
            }
        ]

    async def fake_persist(kind, hubs, **kwargs):
        calls.append(('persist', hubs))

    async def fake_attach(kind, assignments, **kwargs):
        calls.append(
            (
                'attach',
                assignments,
                kwargs['aliases'],
                kwargs['subsumption_edges'],
            )
        )

    async def fake_qualified_meta_hubs(session_factory, kind):
        return {'meta-hub-a'}

    monkeypatch.setattr(
        hub_builder.queries, 'all_source_hubs', fake_source_hubs
    )
    monkeypatch.setattr(
        hub_builder.queries,
        'qualified_meta_hub_uuids',
        fake_qualified_meta_hubs,
    )
    monkeypatch.setattr(hub_builder.queries, 'vector_search', fake_search)
    monkeypatch.setattr(hub_builder.writer, 'persist_hubs', fake_persist)
    monkeypatch.setattr(hub_builder.writer, 'attach_meta_hubs', fake_attach)
    monkeypatch.setattr(
        hub_builder.writer,
        'clear_invalid_meta_hubs',
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        hub_builder.llm,
        'gate',
        lambda max_concurrency: asyncio.Semaphore(1),
    )
    monkeypatch.setattr(
        hub_builder.triplet_hubs,
        'rebuild_meta',
        lambda **kwargs: asyncio.sleep(0),
    )

    result = asyncio.run(
        hub_builder.align_meta_hubs(
            'entity',
            ['source-hub-a', 'source-hub-b'],
            language_model=object(),
            adjudicator=object(),
            synthesizer=object(),
            session_factory=object(),
        )
    )

    assert result == {'aligned': 2, 'new_hubs': 0, 'hierarchies': 0}
    assert calls == [
        ('persist', []),
        (
            'attach',
            [
                {
                    'source_hub': 'source-hub-a',
                    'meta_hub': 'meta-hub-a',
                    'score': 0.95,
                    'decision': 'Merge',
                },
                {
                    'source_hub': 'source-hub-b',
                    'meta_hub': 'meta-hub-a',
                    'score': 0.95,
                    'decision': 'Merge',
                },
            ],
            [{'hub': 'meta-hub-a', 'aliases': ['$G', 'graph']}],
            [],
        ),
    ]


def test_align_meta_requires_two_distinct_sources(monkeypatch):
    async def fake_source_hubs(session_factory, kind, hub_uuids=None):
        return [
            {
                'uuid': 'source-hub-a',
                'name': 'graph',
                'embedding': [1.0, 0.0],
                'source': 'book-a',
            }
        ]

    monkeypatch.setattr(
        hub_builder.queries, 'all_source_hubs', fake_source_hubs
    )

    with pytest.raises(
        RuntimeError, match='requires at least two distinct sources'
    ):
        asyncio.run(
            hub_builder.align_meta_hubs(
                'entity',
                ['source-hub-a'],
                language_model=object(),
                session_factory=object(),
            )
        )
