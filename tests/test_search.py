import asyncio
import logging

import httpx
import pytest

from kms.core import embeddings, models, reranker
from kms.core import search as search_module
from kms.graph import queries


def _query(*texts: str) -> models.SearchQuery:
    return models.SearchQuery(
        parts=[
            models.TextNodeInput(
                local_index=index,
                node_type='query',
                node_text=text,
            )
            for index, text in enumerate(texts)
        ]
    )


def test_search_query_requires_ordered_non_blank_parts():
    with pytest.raises(ValueError, match='at least one'):
        models.SearchQuery(parts=[])
    with pytest.raises(ValueError, match='contiguous'):
        models.SearchQuery(
            parts=[
                models.TextNodeInput(
                    local_index=1, node_type='query', node_text='text'
                )
            ]
        )
    with pytest.raises(ValueError, match='non-blank'):
        _query('   ')


def test_search_rejects_legacy_input_shapes(monkeypatch):
    monkeypatch.setattr(search_module, '_get_language_model', lambda: object())

    async def scenario():
        with pytest.raises(TypeError, match='SearchQuery'):
            await search_module.search(
                'text',
                index_name='node_content',
                text_field='content',
                session_factory=lambda: None,
            )
        with pytest.raises(TypeError, match='SearchQuery'):
            await search_module.search(
                ['text'],
                index_name='node_content',
                text_field='content',
                session_factory=lambda: None,
            )

    asyncio.run(scenario())


def test_search_embeds_rendered_text_and_judges_text_records(monkeypatch):
    class FakeEmbedder:
        def __init__(self):
            self.queries = []

        async def embed_query(self, text):
            self.queries.append(text)
            return [0.1, 0.2]

    class FakeDecomposeJudge:
        def __init__(self, _language_model=None):
            pass

        async def aforward(self, *, parts):
            assert [part.node_text for part in parts] == ['alpha', 'beta']
            return False

    class FakeSearchJudge:
        calls = []

        def __init__(self, _language_model=None):
            pass

        async def aforward(self, *, query, candidates):
            self.calls.append((query, candidates))
            return [search_module.SearchJudgeDecision(index=0, relevant=True)]

    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(embeddings, 'is_configured', lambda: True)
    monkeypatch.setattr(embeddings, 'embedder', lambda: fake_embedder)
    monkeypatch.setattr(search_module, 'DecomposeJudge', FakeDecomposeJudge)
    monkeypatch.setattr(search_module, 'SearchJudge', FakeSearchJudge)
    monkeypatch.setattr(search_module.reranker, 'is_configured', lambda: False)

    async def vector_search(*args, **kwargs):
        return [
            {
                'uuid': 'n1',
                'content': 'image description',
                'type': 'image',
                'image_paths': ['/missing/image.png'],
                'score': 0.8,
            }
        ]

    monkeypatch.setattr(queries, 'vector_search', vector_search)

    async def scenario():
        return await search_module.search(
            _query('alpha', 'beta'),
            index_name='node_content',
            text_field='content',
            session_factory=lambda: None,
            rerank_top_n=5,
            language_model=object(),
        )

    groups = asyncio.run(scenario())
    assert fake_embedder.queries == ['alpha beta']
    assert groups[0].results[0].image_paths == ['/missing/image.png']


def test_local_reranker_orders_scores_and_ties(monkeypatch):
    def handler(request):
        assert request.url.path == '/v1/rerank'
        return httpx.Response(
            200,
            json={
                'results': [
                    {'index': 0, 'relevance_score': 0.5},
                    {'index': 1, 'relevance_score': 0.9},
                    {'index': 2, 'relevance_score': 0.9},
                ]
            },
        )

    monkeypatch.setattr(
        reranker.serve,
        'retrieval_server_manager',
        lambda: _NoopRerankerManager(),
    )
    monkeypatch.setattr(
        reranker,
        '_http_client',
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url='http://local/v1'
        ),
    )
    results = asyncio.run(
        reranker.Reranker(model='fake').rerank(
            'query', ['a', 'b', 'c'], top_n=2
        )
    )
    assert results == [
        {'index': 1, 'relevance_score': 0.9},
        {'index': 2, 'relevance_score': 0.9},
    ]


class _NoopRerankerManager:
    async def aensure_reranker_started(self):
        return None


def test_local_reranker_logs_structured_info(monkeypatch, caplog):
    def handler(request):
        return httpx.Response(
            200,
            json={
                'results': [
                    {'index': 0, 'relevance_score': 0.5},
                    {'index': 1, 'relevance_score': 0.9},
                ]
            },
        )

    monkeypatch.setattr(
        reranker.serve,
        'retrieval_server_manager',
        lambda: _NoopRerankerManager(),
    )
    monkeypatch.setattr(
        reranker,
        '_http_client',
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url='http://local/v1'
        ),
    )
    with caplog.at_level(logging.INFO, logger='kms.core.reranker'):
        results = asyncio.run(
            reranker.Reranker(model='fake').rerank(
                'test query', ['doc a', 'doc b'], top_n=2
            )
        )

    assert results == [
        {'index': 1, 'relevance_score': 0.9},
        {'index': 0, 'relevance_score': 0.5},
    ]
    # The INFO record carries candidate count, selected count, query, documents,
    # results, and duration.
    assert '2 candidates -> 2 selected' in caplog.text
    assert 'query=test query' in caplog.text
    assert 'documents=' in caplog.text
    assert 'results=' in caplog.text
    assert 'duration=' in caplog.text
