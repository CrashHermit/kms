import asyncio
from types import SimpleNamespace

import pytest

from kms import config
from kms.construction import local_procedure_hubs
from kms.core import embeddings, reranker
from kms.graph import local_procedure_hubs as graph_hubs


class _ProcedureEmbedder:
    async def embed(self, texts):
        return [[0.2, 0.8] for _ in texts]


class _ProcedureSynthesizer:
    def __init__(self):
        self.calls = []

    async def aforward(self, surface_forms, descriptions, scope):
        self.calls.append((surface_forms, descriptions, scope))
        return ('Procedure concept', 'A source-local procedure concept.')


class _ProcedureReranker:
    def __init__(self, indexes=(0,)):
        self.indexes = indexes
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [
            {'index': index, 'relevance_score': 1.0}
            for index in self.indexes[:top_n]
        ]


def _procedure_settings(budget=4096, rerank_top_n=1, candidate_limit=32):
    return SimpleNamespace(
        concurrency=SimpleNamespace(max_concurrent_calls=16),
        stages=SimpleNamespace(
            procedure_hubs=SimpleNamespace(
                comparison_token_budget=budget,
                merge_above=0.9,
                separate_below=0.2,
                rerank_top_n=rerank_top_n,
                rerank_candidate_limit=candidate_limit,
            )
        ),
    )


def _procedure_spec():
    return local_procedure_hubs.ProcedureHubSpec(
        graph=None,
        stage_name='procedure_hubs',
        hub_id_factory=lambda records, definition: (
            'procedure:'
            + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: records[0]['source'],
        synthesis_context='procedure scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
    )


def _procedure_record(uuid, embedding, description='procedure description'):
    return {
        'uuid': uuid,
        'name': '',
        'aliases': [],
        'description': description,
        'embedding': embedding,
        'source': 'source',
    }


def _patch_procedure(monkeypatch, settings, enabled=False, client=None):
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(embeddings, 'embedder', _ProcedureEmbedder)
    monkeypatch.setattr(reranker, 'is_configured', lambda: enabled)
    if client is not None:
        monkeypatch.setattr(reranker, 'reranker', lambda: client)


def _build_procedure(records, synthesizer=None):
    return asyncio.run(
        local_procedure_hubs.build_procedure_hubs(
            records,
            source='source',
            spec=_procedure_spec(),
            max_concurrency=3,
            synthesizer=synthesizer or _ProcedureSynthesizer(),
        )
    )


def test_procedure_builder_uses_uuid_pivot_and_assigns_every_record(
    monkeypatch,
):
    _patch_procedure(monkeypatch, _procedure_settings())
    result = _build_procedure(
        [
            _procedure_record('c', [-1.0, 0.0]),
            _procedure_record('b', [0.99, 0.01]),
            _procedure_record('a', [1.0, 0.0]),
        ]
    )
    assert [hub['members'] for hub in result['hubs']] == [['a', 'b'], ['c']]
    assert sorted(item['component'] for item in result['assignments']) == [
        'a',
        'b',
        'c',
    ]
    assert result['diagnostics'] == {
        'records': 3,
        'remaining_records': 0,
        'embedding_comparisons': 2,
        'ambiguous_candidates': 0,
        'reranker_requests': 0,
        'reranker_candidates': 0,
        'reranker_selected': 0,
        'locked_groups': 2,
        'synthesizer_calls': 2,
        'final_hubs': 2,
    }


def test_procedure_synthesis_has_only_sorted_unique_descriptions(monkeypatch):
    _patch_procedure(monkeypatch, _procedure_settings())
    synthesizer = _ProcedureSynthesizer()
    _build_procedure(
        [
            _procedure_record('a', [1.0, 0.0], 'z description'),
            _procedure_record('b', [1.0, 0.0], 'a description'),
            _procedure_record('c', [-1.0, 0.0], 'a description'),
        ],
        synthesizer,
    )
    assert synthesizer.calls == [
        ([], ['a description', 'z description'], 'procedure scope'),
        ([], ['a description'], 'procedure scope'),
    ]


def test_procedure_builder_reranks_description_only_candidates(monkeypatch):
    client = _ProcedureReranker(indexes=(1, 1, 99))
    _patch_procedure(
        monkeypatch,
        _procedure_settings(budget=20, rerank_top_n=3),
        enabled=True,
        client=client,
    )
    result = _build_procedure(
        [
            _procedure_record('a', [1.0, 0.0], 'pivot'),
            _procedure_record('b', [0.5, 0.8660254], 'candidate b'),
            _procedure_record('c', [0.5, -0.8660254], 'candidate c'),
        ]
    )
    assert client.calls == [('pivot', ['candidate b', 'candidate c'], 3)]
    assert result['diagnostics']['reranker_candidates'] == 2
    assert result['diagnostics']['reranker_selected'] == 2
    assert result['hubs'][0]['members'] == ['a', 'c']
    assert result['hubs'][0]['aliases'] == []


def test_procedure_builder_requeues_omitted_candidates(monkeypatch):
    client = _ProcedureReranker(indexes=(0,))
    _patch_procedure(
        monkeypatch,
        _procedure_settings(budget=100, candidate_limit=1),
        enabled=True,
        client=client,
    )
    result = _build_procedure(
        [
            _procedure_record('a', [1.0, 0.0]),
            _procedure_record('b', [0.5, 0.8660254]),
            _procedure_record('c', [0.5, 0.8660254]),
        ]
    )
    assert result['diagnostics']['ambiguous_candidates'] == 2
    assert result['diagnostics']['reranker_candidates'] == 1
    assert sorted(
        member for hub in result['hubs'] for member in hub['members']
    ) == [
        'a',
        'b',
        'c',
    ]


def test_procedure_builder_rejects_source_mismatch_and_missing_embeddings(
    monkeypatch,
):
    _patch_procedure(monkeypatch, _procedure_settings())
    with pytest.raises(ValueError, match='outside source'):
        _build_procedure(
            [_procedure_record('a', [1.0, 0.0]) | {'source': 'other'}]
        )
    with pytest.raises(RuntimeError, match='lack an embedding'):
        _build_procedure([_procedure_record('a', None)])


def test_procedure_builder_rejects_over_budget_pivot_and_candidate(monkeypatch):
    _patch_procedure(
        monkeypatch,
        _procedure_settings(budget=2),
        enabled=True,
        client=_ProcedureReranker(),
    )
    with pytest.raises(ValueError, match='pivot'):
        _build_procedure(
            [
                _procedure_record('a', [1.0, 0.0], 'x' * 20),
                _procedure_record('b', [0.5, 0.866]),
            ]
        )

    _patch_procedure(
        monkeypatch,
        _procedure_settings(budget=7),
        enabled=True,
        client=_ProcedureReranker(),
    )
    with pytest.raises(ValueError, match='candidate'):
        _build_procedure(
            [
                _procedure_record('a', [1.0, 0.0]),
                _procedure_record('b', [0.5, 0.866], 'x' * 20),
            ]
        )


def test_procedure_builder_empty_input_has_complete_diagnostics(monkeypatch):
    _patch_procedure(monkeypatch, _procedure_settings())
    result = _build_procedure([])
    assert result['hubs'] == []
    assert result['assignments'] == []
    assert result['diagnostics']['records'] == 0
    assert result['diagnostics']['remaining_records'] == 0
    assert result['diagnostics']['final_hubs'] == 0


def test_procedure_records_are_normalized_at_query_boundary():
    records = local_procedure_hubs._records(
        [
            {
                'uuid': 'procedure',
                'source': 'source',
                'description': 'description',
                'embedding': (1.0, 0.0),
            }
        ]
    )
    assert records == [
        {
            'uuid': 'procedure',
            'name': '',
            'aliases': [],
            'source': 'source',
            'description': 'description',
            'embedding': [1.0, 0.0],
        }
    ]


def test_procedure_hub_id_factory_uses_procedure_graph_uuid():
    records = [
        _procedure_record('b', [1.0, 0.0]),
        _procedure_record('a', [1.0, 0.0]),
    ]
    hub_id = local_procedure_hubs.source_hub_id_factory(graph_hubs)(records, {})
    assert hub_id == graph_hubs.hub_uuid('source', ['a', 'b'])
