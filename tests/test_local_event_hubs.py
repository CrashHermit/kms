import asyncio
from types import SimpleNamespace

import pytest

from kms import config
from kms.construction import local_event_hubs
from kms.core import embeddings, reranker


class _EventEmbedder:
    async def embed(self, texts):
        return [[0.2, 0.8] for _ in texts]


class _EventSynthesizer:
    async def aforward(self, surface_forms, descriptions, scope):
        return ('Event concept', 'A source-local event concept.')


class _EventReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [{'index': 0, 'relevance_score': 1.0}][:top_n]


def _event_settings(budget=4096, rerank_top_n=1, candidate_limit=32):
    return SimpleNamespace(
        concurrency=SimpleNamespace(max_concurrent_calls=16),
        stages=SimpleNamespace(
            event_hubs=SimpleNamespace(
                comparison_token_budget=budget,
                merge_above=0.9,
                separate_below=0.2,
                rerank_top_n=rerank_top_n,
                rerank_candidate_limit=candidate_limit,
            )
        ),
    )


def _event_spec():
    return local_event_hubs.EventHubSpec(
        graph=None,
        stage_name='event_hubs',
        hub_id_factory=lambda records, definition: (
            'event:' + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: records[0]['source'],
        synthesis_context='event scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
    )


def _event_record(uuid, embedding, name=None):
    return {
        'uuid': uuid,
        'name': name or uuid,
        'aliases': [],
        'description': 'event description',
        'embedding': embedding,
        'source': 'source',
    }


def _patch_event(monkeypatch, settings, enabled=True):
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(embeddings, 'embedder', _EventEmbedder)
    monkeypatch.setattr(reranker, 'is_configured', lambda: enabled)
    if enabled:
        monkeypatch.setattr(reranker, 'reranker', lambda: _EventReranker())


def _build_event(records):
    return asyncio.run(
        local_event_hubs.build_event_hubs(
            records,
            source='source',
            spec=_event_spec(),
            synthesizer=_EventSynthesizer(),
        )
    )


def test_event_builder_uses_sorted_pivot_and_assigns_every_record(monkeypatch):
    _patch_event(monkeypatch, _event_settings(), enabled=False)
    result = _build_event(
        [
            _event_record('c', [-1.0, 0.0]),
            _event_record('b', [0.99, 0.01]),
            _event_record('a', [1.0, 0.0]),
        ]
    )
    assert [hub['members'] for hub in result['hubs']] == [['a', 'b'], ['c']]
    assert sorted(item['component'] for item in result['assignments']) == [
        'a',
        'b',
        'c',
    ]
    assert result['diagnostics']['locked_groups'] == 2


def test_event_builder_reranks_ambiguous_candidates(monkeypatch):
    client = _EventReranker()
    _patch_event(monkeypatch, _event_settings(budget=20), enabled=True)
    monkeypatch.setattr(reranker, 'reranker', lambda: client)
    result = _build_event(
        [
            _event_record('a', [1.0, 0.0]),
            _event_record('b', [0.5, 0.8660254]),
            _event_record('c', [0.5, -0.8660254]),
        ]
    )
    assert len(client.calls) == 1
    assert result['diagnostics']['reranker_candidates'] == 2
    assert result['diagnostics']['reranker_selected'] == 1

    assert len(client.calls) == 1


def test_event_builder_rejects_missing_embeddings(monkeypatch):
    _patch_event(monkeypatch, _event_settings(), enabled=False)
    with pytest.raises(RuntimeError, match='lack an embedding'):
        _build_event([_event_record('a', None)])


def test_event_builder_rejects_over_budget_candidate(monkeypatch):
    _patch_event(monkeypatch, _event_settings(budget=2), enabled=True)
    with pytest.raises(ValueError, match='reranker budget is 2'):
        _build_event(
            [
                _event_record('a', [1.0, 0.0]),
                _event_record('b', [0.5, 0.866], name='x' * 20),
            ]
        )
