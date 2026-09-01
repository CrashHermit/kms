import asyncio
import logging
from types import SimpleNamespace

import pytest

from kms import config
from kms.construction import local_entity_hubs
from kms.core import embeddings, reranker


class _Embedder:
    async def embed(self, texts):
        return [[0.3, 0.4] for _ in texts]


class _Synthesizer:
    def __init__(self):
        self.calls = []

    async def aforward(self, surface_forms, descriptions, scope):
        self.calls.append((list(surface_forms), list(descriptions), scope))
        return ('Canonical entity', 'A source-local canonical entity.')


class _Reranker:
    def __init__(self, indexes):
        self._indexes = indexes
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [
            {'index': index, 'relevance_score': 1.0}
            for index in self._indexes
            if index < len(documents)
        ]


def _settings(*, budget=4096, rerank_top_n=5, candidate_limit=32):
    return SimpleNamespace(
        concurrency=SimpleNamespace(max_concurrent_calls=16),
        stages=SimpleNamespace(
            search=SimpleNamespace(top_k=1),
            entity_hubs=SimpleNamespace(
                comparison_token_budget=budget,
                merge_above=0.9,
                rerank_top_n=rerank_top_n,
                separate_below=0.2,
                rerank_candidate_limit=candidate_limit,
            ),
        ),
    )


def _spec():
    return local_entity_hubs.EntityHubSpec(
        graph=None,
        stage_name='entity_hubs',
        hub_id_factory=lambda records, definition: (
            'hub:' + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: records[0]['source'],
        synthesis_context='Synthesize a source-local semantic hub.',
        all_components=lambda *_: None,
        clear_hubs=lambda *_: None,
        persist_hubs=lambda *_: None,
        rebuild_names=lambda *_: None,
    )


def _record(uuid, embedding, *, name=None, description=''):
    return {
        'uuid': uuid,
        'name': name or uuid,
        'aliases': [],
        'description': description,
        'embedding': embedding,
        'source': 'source',
    }


def _patch_dependencies(monkeypatch, *, settings, reranker_client=None):
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(embeddings, 'embedder', _Embedder)
    monkeypatch.setattr(
        reranker,
        'is_configured',
        lambda: reranker_client is not None,
    )
    if reranker_client is not None:
        monkeypatch.setattr(reranker, 'reranker', lambda: reranker_client)


def _build(records, synthesizer):
    return asyncio.run(
        local_entity_hubs.build_entity_hubs(
            records,
            source='source',
            spec=_spec(),
            synthesizer=synthesizer,
        )
    )


def test_source_builder_compares_every_unassigned_entity(monkeypatch):
    _patch_dependencies(monkeypatch, settings=_settings())
    synthesizer = _Synthesizer()

    result = _build(
        [
            _record('c', [-1.0, 0.0]),
            _record('b', [0.99, 0.01]),
            _record('a', [1.0, 0.0]),
        ],
        synthesizer,
    )

    assert result['diagnostics']['embedding_comparisons'] == 2
    assert result['diagnostics']['locked_groups'] == 2
    assert [hub['members'] for hub in result['hubs']] == [['a', 'b'], ['c']]
    assert result['assignments'] == [
        {'component': 'a', 'hub': 'hub:a|b'},
        {'component': 'b', 'hub': 'hub:a|b'},
        {'component': 'c', 'hub': 'hub:c'},
    ]
    assert len(synthesizer.calls) == 2


def test_source_builder_reranks_every_ambiguous_entity_in_token_batches(
    monkeypatch,
):
    reranker_client = _Reranker([0])
    _patch_dependencies(
        monkeypatch,
        settings=_settings(budget=2, rerank_top_n=1),
        reranker_client=reranker_client,
    )

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
            _record('c', [0.5, -0.8660254]),
            _record('d', [0.5, 0.8660254]),
        ],
        _Synthesizer(),
    )

    assert [call[1] for call in reranker_client.calls] == [['b'], ['c'], ['d']]
    assert [call[2] for call in reranker_client.calls] == [1, 1, 1]
    assert result['diagnostics'] == {
        'records': 4,
        'remaining_records': 0,
        'embedding_comparisons': 3,
        'ambiguous_candidates': 3,
        'reranker_requests': 3,
        'reranker_candidates': 3,
        'reranker_selected': 3,
        'locked_groups': 1,
        'synthesizer_calls': 1,
        'final_hubs': 1,
    }
    assert result['hubs'][0]['members'] == ['a', 'b', 'c', 'd']


def test_source_builder_admits_only_top_embedding_candidates(monkeypatch):
    reranker_client = _Reranker([0])
    _patch_dependencies(
        monkeypatch,
        settings=_settings(candidate_limit=1),
        reranker_client=reranker_client,
    )

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
            _record('c', [0.5, 0.8660254]),
        ],
        _Synthesizer(),
    )

    assert all(len(call[1]) <= 1 for call in reranker_client.calls)
    assert result['diagnostics']['ambiguous_candidates'] == 2
    assert result['diagnostics']['reranker_candidates'] == 1


def test_source_builder_leaves_unselected_ambiguous_entities_unassigned(
    monkeypatch,
):
    _patch_dependencies(monkeypatch, settings=_settings())

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
            _record('c', [-1.0, 0.0]),
        ],
        _Synthesizer(),
    )

    assert [hub['members'] for hub in result['hubs']] == [['a'], ['b'], ['c']]
    assert result['diagnostics']['reranker_requests'] == 0
    assert result['diagnostics']['remaining_records'] == 0


def test_source_builder_rejects_an_over_budget_reranker_input(monkeypatch):
    _patch_dependencies(
        monkeypatch,
        settings=_settings(budget=2),
        reranker_client=_Reranker([0]),
    )

    with pytest.raises(ValueError, match="candidate 'b'.*reranker budget is 2"):
        _build(
            [
                _record('a', [1.0, 0.0]),
                _record('b', [0.5, 0.8660254], name='x' * 20),
            ],
            _Synthesizer(),
        )


def test_source_builder_records_empty_diagnostics(monkeypatch):
    _patch_dependencies(monkeypatch, settings=_settings())

    result = _build([], _Synthesizer())

    assert result['diagnostics'] == {
        'records': 0,
        'remaining_records': 0,
        'embedding_comparisons': 0,
        'ambiguous_candidates': 0,
        'reranker_requests': 0,
        'reranker_candidates': 0,
        'reranker_selected': 0,
        'locked_groups': 0,
        'synthesizer_calls': 0,
        'final_hubs': 0,
    }


def test_source_builder_logs_cumulative_diagnostics(monkeypatch, caplog):
    _patch_dependencies(monkeypatch, settings=_settings())

    with caplog.at_level(
        logging.INFO, logger='kms.construction.local_entity_hubs'
    ):
        _build(
            [
                _record('a', [1.0, 0.0]),
                _record('b', [-1.0, 0.0]),
            ],
            _Synthesizer(),
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == 'kms.construction.local_entity_hubs'
    ]
    diagnostics = [
        message
        for message in messages
        if message.startswith('entity hub diagnostics:')
    ]
    assert len(diagnostics) >= 4
    assert any("'group': 1" in message for message in diagnostics)
    assert any("'remaining_records': 0" in message for message in diagnostics)
