import asyncio

import dspy
from PIL import Image

from kms.core import content, embeddings
from kms.core import search as search_module
from kms.graph import queries


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _session_factory():
    return _FakeSession()


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


class _FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    async def rerank(self, *, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [
            {'index': index, 'relevance_score': 1.0 - index * 0.1}
            for index in range(len(documents))
        ]


def _async_return(value):
    async def _call(*args, **kwargs):
        return value

    return _call


def _candidate(uuid_value: str) -> dict:
    return {
        'uuid': uuid_value,
        'description': f'desc-{uuid_value}',
        'score': 0.5,
    }


def _install_fakes(monkeypatch, *, vector_search):
    monkeypatch.setattr(embeddings, 'is_configured', lambda: True)
    monkeypatch.setattr(search_module, '_get_language_model', lambda: object())
    fake_embedder = _FakeEmbedder()
    monkeypatch.setattr(embeddings, 'embedder', lambda: fake_embedder)
    monkeypatch.setattr(
        search_module.SearchJudge, 'aforward', _async_return([])
    )
    monkeypatch.setattr(search_module.reranker, 'is_configured', lambda: False)
    monkeypatch.setattr(queries, 'vector_search', vector_search)
    return fake_embedder


def test_raw_path_returns_single_group(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a'), _candidate('b')]

    fake_embedder = _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )

    async def scenario():
        return await search_module.search(
            'subgraph',
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    groups = asyncio.run(scenario())
    assert len(groups) == 1
    assert groups[0].sub_query == 'subgraph'
    assert [result.properties['uuid'] for result in groups[0].results] == [
        'a',
        'b',
    ]
    assert fake_embedder.calls == [[content.Content.from_text('subgraph')]]


def test_source_filter_is_forwarded_to_vector_search(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        assert source == 'book-a'
        return [_candidate('a')]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )

    async def scenario():
        return await search_module.search(
            'subgraph',
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            source='book-a',
        )

    groups = asyncio.run(scenario())
    assert groups[0].results[0].properties['uuid'] == 'a'


def test_decomposed_path_slices_query_parts(monkeypatch):
    calls = {'count': 0}

    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        calls['count'] += 1
        return [_candidate(str(calls['count']))]

    fake_embedder = _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(True)
    )
    monkeypatch.setattr(
        search_module.QueryDecomposer,
        'aforward',
        _async_return(
            [
                search_module.SubQueryPlan(label='first', part_indices=[0, 1]),
                search_module.SubQueryPlan(label='third', part_indices=[2]),
            ]
        ),
    )

    async def scenario():
        return await search_module.search(
            ['alpha', 'beta', 'gamma'],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    groups = asyncio.run(scenario())
    assert [group.sub_query for group in groups] == ['first', 'third']
    assert [
        result.properties['uuid']
        for group in groups
        for result in group.results
    ] == ['1', '2']
    assert fake_embedder.calls == [
        [content.Content.from_parts(['alpha', 'beta'])],
        [content.Content.from_text('gamma')],
    ]


def test_image_query_embeds_multimodal_parts(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a')]

    fake_embedder = _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )
    image = dspy.Image(url='data:image/png;base64,AAAA')

    async def scenario():
        return await search_module.search(
            ['What is this?', image],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    groups = asyncio.run(scenario())
    assert len(groups) == 1
    assert len(fake_embedder.calls) == 1
    embedding_input = fake_embedder.calls[0][0]
    assert isinstance(embedding_input, content.Content)
    assert embedding_input.parts[1].image is image


def test_image_query_rerank_documents_carry_image(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a'), _candidate('b')]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )
    fake_reranker = _FakeReranker()
    monkeypatch.setattr(search_module.reranker, 'is_configured', lambda: True)
    monkeypatch.setattr(
        search_module.reranker, 'reranker', lambda: fake_reranker
    )
    image = dspy.Image(url='data:image/png;base64,AAAA')

    async def scenario():
        return await search_module.search(
            ['What is this?', image],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    asyncio.run(scenario())
    assert len(fake_reranker.calls) == 1
    query_text, documents, top_n = fake_reranker.calls[0]
    assert query_text == 'What is this? [image]'
    assert top_n == 5
    assert documents == [
        {'text': 'desc-a', 'image': 'data:image/png;base64,AAAA'},
        {'text': 'desc-b', 'image': 'data:image/png;base64,AAAA'},
    ]


def test_image_query_decompose_judge_receives_images(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a')]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    calls = {}

    async def _decompose(self, parts):
        calls['parts'] = parts
        return False

    monkeypatch.setattr(search_module.DecomposeJudge, 'aforward', _decompose)
    image = dspy.Image(url='data:image/png;base64,AAAA')

    async def scenario():
        return await search_module.search(
            ['What is this?', image],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    asyncio.run(scenario())
    assert calls['parts'] == content.Content.from_parts(
        ['What is this?', image]
    )


def test_image_query_decomposer_receives_images(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a')]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(True)
    )
    calls = {}

    async def _decompose(self, parts):
        calls['parts'] = parts
        return [search_module.SubQueryPlan(label='all', part_indices=[0, 1])]

    monkeypatch.setattr(search_module.QueryDecomposer, 'aforward', _decompose)
    image = dspy.Image(url='data:image/png;base64,AAAA')

    async def scenario():
        return await search_module.search(
            ['What is this?', image],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    asyncio.run(scenario())
    assert calls['parts'] == content.Content.from_parts(
        ['What is this?', image]
    )


def test_image_query_relevance_judge_receives_images(monkeypatch):
    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [_candidate('a')]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )
    calls = {}

    async def _judge(self, parts, candidates):
        calls['parts'] = parts
        calls['candidates'] = candidates
        return []

    monkeypatch.setattr(search_module.SearchJudge, 'aforward', _judge)
    image = dspy.Image(url='data:image/png;base64,AAAA')

    async def scenario():
        return await search_module.search(
            ['What is this?', image],
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    asyncio.run(scenario())
    assert calls['parts'] == content.Content.from_parts(
        ['What is this?', image]
    )
    assert calls['candidates'].content == content.Content(
        parts=[
            content.TextPart(text='Candidate 0:'),
            content.TextPart(text='desc-a'),
        ]
    )


class _FakeClient:
    def __init__(self) -> None:
        self.posts = []

    async def post(self, url, *, json):
        self.posts.append((url, json))
        return _FakeResponse()


class _FakeResponse:
    status_code = 200
    text = ''

    def json(self):
        return {
            'data': [
                {'embedding': [0.1, 0.2]},
                {'embedding': [0.1, 0.2]},
            ]
        }


def test_embed_batch_wraps_content_per_input(monkeypatch):
    embedder = embeddings.Embedder(api_key='test', dimension=2)
    fake_client = _FakeClient()

    async def _client_for():
        return fake_client

    monkeypatch.setattr(embedder, '_client_for', _client_for)
    contents = [
        content.Content.from_text('a'),
        content.Content.from_text('b'),
    ]
    vectors = asyncio.run(embedder._embed_batch(contents))
    url, payload = fake_client.posts[0]
    assert url == 'https://api.voyageai.com/v1/multimodalembeddings'
    assert payload == {
        'model': 'voyage-multimodal-3.5',
        'inputs': [
            {'content': [{'type': 'text', 'text': 'a'}]},
            {'content': [{'type': 'text', 'text': 'b'}]},
        ],
    }
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]


def test_image_candidate_reaches_judge_as_image(monkeypatch, tmp_path):
    image_file = tmp_path / 'fig.png'
    Image.new('RGB', (10, 10), (0, 255, 0)).save(image_file)

    async def _vector_search(
        session_factory, *, index_name, query_embedding, top_k, source
    ):
        return [{**_candidate('a'), 'image_path': str(image_file)}]

    _install_fakes(monkeypatch, vector_search=_vector_search)
    monkeypatch.setattr(
        search_module.DecomposeJudge, 'aforward', _async_return(False)
    )
    calls = {}

    async def _judge(self, parts, candidates):
        calls['candidates'] = candidates
        return []

    monkeypatch.setattr(search_module.SearchJudge, 'aforward', _judge)

    async def scenario():
        return await search_module.search(
            'subgraph',
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=_session_factory,
            top_k=10,
        )

    asyncio.run(scenario())
    parts = calls['candidates'].content.parts
    assert parts[0] == content.TextPart(text='Candidate 0:')
    assert parts[1] == content.TextPart(text='desc-a')
    assert isinstance(parts[2], content.ImagePart)
