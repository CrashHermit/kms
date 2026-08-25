import asyncio

import httpx
import pytest

from kms.core import embeddings, models
from kms.graph import nodes


class _Embedder:
    def __init__(self):
        self.inputs = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        return [[float(index + 1)] for index in range(len(texts))]


def test_source_nodes_embed_descriptions_without_opening_assets(monkeypatch):
    fake = _Embedder()
    monkeypatch.setattr(embeddings, 'embedder', lambda: fake)
    source_nodes = [
        models.SourceNode(type='paragraph', content='A theorem.'),
        models.SourceNode(
            type='image',
            content='A diagram of the theorem.',
            assets=[models.VisualAsset(path='/missing/figure.png')],
        ),
        models.SourceNode(type='paragraph', content='The proof follows.'),
        models.SourceNode(type='paragraph', content='   '),
    ]
    vectors = asyncio.run(embeddings.embed_source_nodes(source_nodes))
    assert vectors == [[1.0], [2.0], [3.0], None]
    assert fake.inputs == ['A theorem.', 'A diagram of the theorem.', 'The proof follows.']


def test_empty_source_nodes_return_position_preserving_nones(monkeypatch):
    fake = _Embedder()
    monkeypatch.setattr(embeddings, 'embedder', lambda: fake)
    assert asyncio.run(
        embeddings.embed_source_nodes([models.SourceNode(type='image', assets=[])])
    ) == [None]
    assert fake.inputs == []


def test_node_properties_use_source_node_embedding():
    node = models.SourceNode(uuid='node-1', type='paragraph', content='text', embedding=[0.25, 0.75])
    assert nodes.node_properties(node, 'book.pdf')['embedding'] == [0.25, 0.75]


def test_local_embedder_batches_and_preserves_response_order(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.read())
        return httpx.Response(
            200,
            json={
                'data': [
                    {'index': 1, 'embedding': [3.0, 4.0]},
                    {'index': 0, 'embedding': [1.0, 2.0]},
                ]
            },
        )

    monkeypatch.setattr(embeddings.serve, 'retrieval_server_manager', lambda: _NoopManager())
    monkeypatch.setattr(
        embeddings,
        '_http_client',
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://local/v1'),
    )
    instance = embeddings.Embedder(model='fake', batch_size=2, dimension=2)
    assert asyncio.run(instance.embed(['a', 'b'])) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls


def test_local_embedder_rejects_dimension_mismatch(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0]}]})

    monkeypatch.setattr(embeddings.serve, 'retrieval_server_manager', lambda: _NoopManager())
    monkeypatch.setattr(
        embeddings,
        '_http_client',
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://local/v1'),
    )
    with pytest.raises(RuntimeError, match='dimension mismatch'):
        asyncio.run(embeddings.Embedder(dimension=2).embed(['text']))


class _NoopManager:
    async def aensure_embedding_started(self):
        return None
