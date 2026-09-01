import asyncio
from types import SimpleNamespace

import pytest

from kms import config
from kms.construction import local_statement_hubs
from kms.core import embeddings, models, reranker
from kms.graph import local_statement_hubs as graph_hubs


class _StatementEmbedder:
    async def embed(self, texts):
        return [[0.2, 0.8] for _ in texts]


class _StatementSynthesizer:
    def __init__(self):
        self.calls = []

    async def aforward(self, surface_forms, descriptions, scope):
        self.calls.append((list(surface_forms), list(descriptions), scope))
        return ('Statement concept', 'A canonical statement description.')


class _StatementReranker:
    def __init__(self, indexes=None):
        self.indexes = [0] if indexes is None else list(indexes)
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [
            {'index': index, 'relevance_score': 1.0} for index in self.indexes
        ][:top_n]


def _settings(*, budget=4096, top_n=5, candidate_limit=32):
    return SimpleNamespace(
        concurrency=SimpleNamespace(max_concurrent_calls=16),
        stages=SimpleNamespace(
            statement_hubs=SimpleNamespace(
                comparison_token_budget=budget,
                merge_above=0.9,
                separate_below=0.2,
                rerank_top_n=top_n,
                rerank_candidate_limit=candidate_limit,
            )
        ),
    )


def _spec():
    return local_statement_hubs.StatementHubSpec(
        graph=None,
        stage_name='statement_hubs',
        hub_id_factory=lambda records, definition: (
            'hub:' + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: records[0]['source'],
        synthesis_context='statement scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
    )


def _record(uuid, embedding, *, description='statement description', name=None):
    return {
        'uuid': uuid,
        'name': name or f'name-{uuid}',
        'aliases': [f'alias-{uuid}'],
        'description': description,
        'embedding': embedding,
        'source': 'source',
    }


def _patch(monkeypatch, settings, client=None):
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(embeddings, 'embedder', _StatementEmbedder)
    monkeypatch.setattr(reranker, 'is_configured', lambda: client is not None)
    if client is not None:
        monkeypatch.setattr(reranker, 'reranker', lambda: client)


def _build(records, synthesizer=None):
    return asyncio.run(
        local_statement_hubs.build_statement_hubs(
            records,
            source='source',
            spec=_spec(),
            synthesizer=synthesizer or _StatementSynthesizer(),
        )
    )


def test_statement_builder_uses_sorted_pivot_and_assigns_every_record(
    monkeypatch,
):
    _patch(monkeypatch, _settings())
    synthesizer = _StatementSynthesizer()

    result = _build(
        [
            _record('c', [-1.0, 0.0]),
            _record('b', [0.99, 0.01]),
            _record('a', [1.0, 0.0]),
        ],
        synthesizer,
    )

    assert [hub['members'] for hub in result['hubs']] == [['a', 'b'], ['c']]
    assert result['assignments'] == [
        {'component': 'a', 'hub': 'hub:a|b'},
        {'component': 'b', 'hub': 'hub:a|b'},
        {'component': 'c', 'hub': 'hub:c'},
    ]
    assert result['diagnostics']['embedding_comparisons'] == 2
    assert result['diagnostics']['locked_groups'] == 2
    assert len(synthesizer.calls) == 2


def test_statement_synthesis_uses_empty_forms_and_sorted_unique_descriptions(
    monkeypatch,
):
    _patch(monkeypatch, _settings())
    synthesizer = _StatementSynthesizer()

    _build(
        [
            _record('b', [1.0, 0.0], description='Zed', name='semantic name'),
            _record('a', [1.0, 0.0], description='Alpha', name='other name'),
            _record('c', [1.0, 0.0], description='Zed', name='third name'),
        ],
        synthesizer,
    )

    assert synthesizer.calls == [([], ['Alpha', 'Zed'], 'statement scope')]


def test_statement_builder_reranks_ambiguous_candidates_in_local_batches(
    monkeypatch,
):
    client = _StatementReranker([0])
    _patch(monkeypatch, _settings(budget=2, top_n=1), client)

    result = _build(
        [
            _record('a', [1.0, 0.0], description='a'),
            _record('b', [0.5, 0.8660254], description='b'),
            _record('c', [0.5, -0.8660254], description='c'),
            _record('d', [0.5, 0.8660254], description='d'),
        ]
    )

    assert [call[1] for call in client.calls] == [['b'], ['c'], ['d']]
    assert result['hubs'][0]['members'] == ['a', 'b', 'c', 'd']
    assert result['diagnostics']['reranker_requests'] == 3
    assert result['diagnostics']['reranker_candidates'] == 3


def test_statement_builder_admits_only_top_ambiguous_candidates(monkeypatch):
    client = _StatementReranker([0])
    _patch(monkeypatch, _settings(candidate_limit=1), client)

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
            _record('c', [0.5, 0.8660254]),
        ]
    )

    assert all(len(call[1]) <= 1 for call in client.calls)
    assert result['diagnostics']['ambiguous_candidates'] == 2
    assert result['diagnostics']['reranker_candidates'] == 1


def test_statement_builder_requeues_omitted_ambiguous_candidates(monkeypatch):
    client = _StatementReranker([])
    _patch(monkeypatch, _settings(candidate_limit=1), client)

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
            _record('c', [0.5, 0.8660254]),
        ]
    )

    assert [hub['members'] for hub in result['hubs']] == [
        ['a'],
        ['b', 'c'],
    ]
    assert result['diagnostics']['remaining_records'] == 0


def test_statement_builder_disabled_reranker_leaves_ambiguous_singletons(
    monkeypatch,
):
    _patch(monkeypatch, _settings(), None)

    result = _build(
        [
            _record('a', [1.0, 0.0]),
            _record('b', [0.5, 0.8660254]),
        ]
    )

    assert [hub['members'] for hub in result['hubs']] == [['a'], ['b']]
    assert result['diagnostics']['reranker_requests'] == 0


def test_statement_builder_validates_batch_indexes_and_deduplicates(
    monkeypatch,
):
    client = _StatementReranker([0, 8, 0])
    _patch(monkeypatch, _settings(budget=2, top_n=3), client)

    result = _build(
        [
            _record('a', [1.0, 0.0], description='a'),
            _record('b', [0.5, 0.8660254], description='b'),
            _record('c', [0.5, 0.8660254], description='c'),
        ]
    )

    assert result['hubs'][0]['members'] == ['a', 'b', 'c']
    assert result['diagnostics']['reranker_selected'] == 4


def test_statement_builder_rejects_source_mismatch_and_missing_embeddings(
    monkeypatch,
):
    _patch(monkeypatch, _settings(), None)
    outside = _record('a', [1.0, 0.0])
    outside['source'] = 'other'
    with pytest.raises(ValueError, match='outside source'):
        _build([outside])
    with pytest.raises(RuntimeError, match='lack an embedding'):
        _build([_record('a', None)])


def test_statement_builder_rejects_over_budget_pivot_and_candidate(monkeypatch):
    client = _StatementReranker([0])
    _patch(monkeypatch, _settings(budget=2), client)
    with pytest.raises(ValueError, match='pivot.*reranker budget is 2'):
        _build(
            [
                _record('a', [1.0, 0.0], description='x' * 8),
                _record('b', [0.5, 0.8660254], description='b'),
            ]
        )

    _patch(monkeypatch, _settings(budget=2), client)
    with pytest.raises(ValueError, match="candidate 'b'.*reranker budget is 2"):
        _build(
            [
                _record('a', [1.0, 0.0], description='a'),
                _record('b', [0.5, 0.8660254], description='x' * 20),
            ]
        )


def test_statement_builder_empty_input_has_exact_diagnostics(monkeypatch):
    _patch(monkeypatch, _settings(), None)
    result = _build([])
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


def test_statement_records_are_normalized_at_query_boundary():
    records = local_statement_hubs._records(
        [
            {
                'uuid': 'statement',
                'source': 'source',
                'description': 'description',
                'embedding': (1.0, 0.0),
            }
        ]
    )
    assert records == [
        {
            'uuid': 'statement',
            'name': '',
            'aliases': [],
            'source': 'source',
            'description': 'description',
            'embedding': [1.0, 0.0],
        }
    ]


def test_statement_synthesizer_has_strict_structured_boundaries():
    encoded = local_statement_hubs.StatementHubSynthesizer.encode(
        None, [], ['A claim.', 'A second claim.'], 'scope'
    )
    assert encoded == {
        'request': models.HubSynthesisInput(
            surface_forms=[],
            descriptions=['A claim.', 'A second claim.'],
            scope='scope',
        )
    }
    synthesizer = local_statement_hubs.StatementHubSynthesizer.__new__(
        local_statement_hubs.StatementHubSynthesizer
    )
    result = local_statement_hubs.StatementHubDefinition(
        canonical_name='Concept', description='Meaning.'
    )
    assert synthesizer.decode(SimpleNamespace(result=result)) == (
        'Concept',
        'Meaning.',
    )
    with pytest.raises(TypeError, match='StatementHubDefinition'):
        synthesizer.decode(
            SimpleNamespace(result={'canonical_name': 'Concept'})
        )
    with pytest.raises(ValueError, match='canonical_name'):
        synthesizer.decode(
            SimpleNamespace(
                result=local_statement_hubs.StatementHubDefinition(
                    canonical_name='', description='Meaning.'
                )
            )
        )


def test_statement_node_normalizes_records_and_updates_bundle(monkeypatch):
    captured = {}
    hubs = [
        {
            'uuid': 'statement-hub',
            'source': 'source',
            'canonical_name': 'claim',
            'aliases': [],
            'description': 'A claim.',
            'embedding': [0.2, 0.8],
            'members': ['statement-record'],
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
            'assignments': [],
            'diagnostics': diagnostics,
            'hubs': hubs,
            'records': 1,
        }

    monkeypatch.setattr(
        local_statement_hubs, 'build_statement_hubs', fake_build
    )
    synthesizer = object()
    record = models.StatementHubRecord(
        uuid='statement-record',
        source='source',
        description='A statement mention.',
        embedding=[1.0, 0.0],
    )
    result = asyncio.run(
        local_statement_hubs.StatementHubNode(synthesizer).run(
            {
                'source': models.Source(key='source'),
                'statement_hub_records': [record],
            }
        )
    )

    assert captured['records'] == [
        {
            'uuid': 'statement-record',
            'source': 'source',
            'name': '',
            'aliases': [],
            'description': 'A statement mention.',
            'embedding': [1.0, 0.0],
        }
    ]
    assert captured['source'] == 'source'
    assert captured['spec'].graph is graph_hubs
    assert captured['synthesizer'] is synthesizer
    assert result['statement_hubs_created'] == 1
    assert result['statements_clustered'] == 1
    assert result['statement_hub_diagnostics'] == diagnostics
    assert result['statement_hubs'] == hubs
    assert result['construction_bundle'].statement_hubs == hubs
    assert not hasattr(
        result['construction_bundle'], 'statement_hub_assignments'
    )


def test_statement_rebuild_queries_then_clears_then_persists(monkeypatch):
    events = []
    rows = [_record('a', [1.0, 0.0])]
    result = {
        'assignments': [],
        'clusters': 1,
        'diagnostics': {},
        'hubs': [{'uuid': 'hub'}],
        'records': 1,
    }

    async def query(session_factory, source):
        events.append(('query', source))
        return rows

    async def build(records, *, source, spec, max_concurrency, synthesizer):
        events.append(('build', records, source, max_concurrency, synthesizer))
        return result

    async def clear(source, *, session_factory):
        events.append(('clear', source))

    async def persist(hubs, *, session_factory):
        events.append(('persist', hubs))

    monkeypatch.setattr(
        local_statement_hubs.queries, 'statement_hub_items', query
    )
    monkeypatch.setattr(local_statement_hubs, 'build_statement_hubs', build)
    monkeypatch.setattr(
        local_statement_hubs.writer, 'clear_statement_hubs', clear
    )
    monkeypatch.setattr(
        local_statement_hubs.writer, 'persist_statement_hubs', persist
    )
    synthesizer = object()
    rebuilt = asyncio.run(
        local_statement_hubs.rebuild(
            'source',
            session_factory=object(),
            synthesizer=synthesizer,
            max_concurrency=3,
        )
    )

    assert [event[0] for event in events] == [
        'query',
        'build',
        'clear',
        'persist',
    ]
    assert events[1][3:] == (3, synthesizer)
    assert rebuilt == {'clusters': 1, 'records': 1}


def test_statement_source_hub_uuid_preserves_specialized_graph_factory():
    records = [_record('b', [1.0, 0.0]), _record('a', [1.0, 0.0])]
    expected = graph_hubs.hub_uuid('source', ['a', 'b'])
    assert (
        local_statement_hubs.source_hub_id_factory(graph_hubs)(records, {})
        == expected
    )
