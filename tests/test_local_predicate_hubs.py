import asyncio
from types import SimpleNamespace

import pytest

from kms import config
from kms.construction import local_predicate_hubs
from kms.core import embeddings, models, reranker


class _PredicateEmbedder:
    async def embed(self, texts):
        return [[0.4, 0.6] for _ in texts]


class _PredicateSynthesizer:
    async def aforward(self, surface_forms, descriptions, scope):
        return ('Predicate concept', 'A source-local predicate concept.')


class _PredicateReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [{'index': 0, 'relevance_score': 1.0}][:top_n]


def _settings(budget=4096, top_n=1, limit=32):
    return SimpleNamespace(
        concurrency=SimpleNamespace(max_concurrent_calls=16),
        stages=SimpleNamespace(
            predicate_hubs=SimpleNamespace(
                comparison_token_budget=budget,
                merge_above=0.9,
                separate_below=0.2,
                rerank_top_n=top_n,
                rerank_candidate_limit=limit,
            )
        ),
    )


def _spec():
    return local_predicate_hubs.PredicateHubSpec(
        graph=None,
        stage_name='predicate_hubs',
        hub_id_factory=lambda records, definition: (
            'predicate:'
            + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: records[0]['source'],
        synthesis_context='predicate scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
        rebuild_names=None,
    )


def _record(uuid, vector, name=None):
    return {
        'uuid': uuid,
        'name': name or uuid,
        'aliases': [],
        'description': 'predicate description',
        'embedding': vector,
        'source': 'source',
    }


def _patch(monkeypatch, settings, client=None):
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(embeddings, 'embedder', _PredicateEmbedder)
    monkeypatch.setattr(reranker, 'is_configured', lambda: client is not None)
    if client is not None:
        monkeypatch.setattr(reranker, 'reranker', lambda: client)


def _build(records):
    return asyncio.run(
        local_predicate_hubs.build_predicate_hubs(
            records,
            source='source',
            spec=_spec(),
            synthesizer=_PredicateSynthesizer(),
        )
    )


def test_predicate_builder_assigns_sorted_pivot_cluster(monkeypatch):
    _patch(monkeypatch, _settings())
    result = _build(
        [
            _record('c', [-1.0, 0.0]),
            _record('b', [0.99, 0.01]),
            _record('a', [1.0, 0.0]),
        ]
    )
    assert [hub['members'] for hub in result['hubs']] == [['a', 'b'], ['c']]
    assert sorted(item['component'] for item in result['assignments']) == [
        'a',
        'b',
        'c',
    ]


def test_predicate_builder_respects_candidate_limit_and_top_n(monkeypatch):
    client = _PredicateReranker()
    _patch(monkeypatch, _settings(budget=20, top_n=1, limit=1), client)
    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.866]),
            _record('c', [0.5, 0.866]),
        ]
    )
    assert len(client.calls) == 1
    assert len(client.calls[0][1]) == 1
    assert client.calls[0][2] == 1
    assert result['diagnostics']['ambiguous_candidates'] == 2


def test_predicate_builder_disabled_reranker_leaves_singletons(monkeypatch):
    _patch(monkeypatch, _settings(), None)
    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.866]),
        ]
    )
    assert [hub['members'] for hub in result['hubs']] == [['a'], ['b']]
    assert result['diagnostics']['reranker_requests'] == 0


def test_predicate_builder_rejects_invalid_inputs(monkeypatch):
    _patch(monkeypatch, _settings(), None)
    with pytest.raises(RuntimeError, match='lack an embedding'):
        _build([_record('a', None)])
    _patch(monkeypatch, _settings(budget=2), _PredicateReranker())
    with pytest.raises(ValueError, match='reranker budget is 2'):
        _build([_record('a', [1.0, 0.0]), _record('b', [0.5, 0.866], 'x' * 20)])


def test_predicate_hub_node_normalizes_components_and_updates_bundle(
    monkeypatch,
):
    captured = {}
    assignments = [{'component': 'predicate-component', 'hub': 'predicate-hub'}]
    hubs = [
        {
            'uuid': 'predicate-hub',
            'source': 'source',
            'canonical_name': 'relates',
            'aliases': ['relates'],
            'description': 'A relation.',
            'embedding': [0.3, 0.7],
            'members': ['predicate-component'],
        }
    ]
    diagnostics = {'records': 1, 'final_hubs': 1}

    async def fake_build(records, *, source, spec, synthesizer):
        captured.update(
            records=records,
            source=source,
            spec=spec,
            synthesizer=synthesizer,
        )
        return {
            'assignments': assignments,
            'diagnostics': diagnostics,
            'hubs': hubs,
        }

    monkeypatch.setattr(
        local_predicate_hubs, 'build_predicate_hubs', fake_build
    )
    synthesizer = object()
    component = models.HubComponent(
        uuid='predicate-component',
        source='source',
        node_id=3,
        name='relates',
        description='A relation mention.',
        embedding=[1.0, 0.0],
    )

    result = asyncio.run(
        local_predicate_hubs.PredicateHubNode(synthesizer).run(
            {
                'source': models.Source(key='source'),
                'predicate_hub_components': [component],
            }
        )
    )

    assert captured['records'] == [
        {
            'uuid': 'predicate-component',
            'name': 'relates',
            'description': 'A relation mention.',
            'embedding': [1.0, 0.0],
            'source': 'source',
            'aliases': [],
        }
    ]
    assert captured['source'] == 'source'
    assert captured['spec'].graph is not None
    assert captured['synthesizer'] is synthesizer
    assert result['predicate_hub_assignments'] == assignments
    assert result['predicate_hub_diagnostics'] == diagnostics
    assert result['predicate_hub_records'] == hubs
    assert (
        result['construction_bundle'].predicate_hub_assignments == assignments
    )
    assert result['construction_bundle'].predicate_hub_records == hubs
