"""Behavioral tests for the IONOS multimodal reranker."""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from kms.config.reranking import RerankingSettings
from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.inference.reranking.budget import IonosRerankingBudget
from kms.inference.reranking.ionos import IonosReranker


class _ImageMetadata:
    byte_count = 5


class _ImageMetadataCache:
    def measure(self, image: ContentImage) -> _ImageMetadata:
        return _ImageMetadata()


class _Budget:
    image_metadata = _ImageMetadataCache()

    def count_pairs(
        self,
        query: MultimodalContent,
        documents: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
    ) -> list[int]:
        return [len(document.text or '') + 1 for document in documents]


def _settings(**overrides: object) -> RerankingSettings:
    return RerankingSettings(api_key=SecretStr('test-key'), **overrides)


def test_ionos_wire_shape_and_original_index_mapping() -> None:
    async def run() -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    'results': [
                        {'index': 1, 'relevance_score': 0.9},
                        {'index': 0, 'relevance_score': 0.2},
                    ]
                },
            )

        async def encode(images: list[ContentImage]) -> list[str]:
            return ['data:image/png;base64,aW1hZ2U=' for _ in images]

        import kms.inference.reranking.ionos as adapter

        original = adapter.encode_content_images
        adapter.encode_content_images = encode
        try:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                reranker = IonosReranker(
                    _settings(),
                    _Budget(),
                    client,  # type: ignore[arg-type]
                )
                result = await reranker.rerank(
                    MultimodalContent(text='question', image_positions=[0]),
                    [MultimodalContent(text='a'), MultimodalContent(text='b')],
                    content_images=[ContentImage(('asset',), b'image')],
                )
        finally:
            adapter.encode_content_images = original
        assert [item['index'] for item in result] == [1, 0]
        body = json.loads(requests[0].content)
        assert body['model'] == 'Qwen/Qwen3-VL-Reranker-8B'
        assert body['query'][0] == {'type': 'text', 'text': 'question'}
        assert body['query'][1]['type'] == 'image_url'
        assert body['documents'][0]['content'][0]['text'] == 'a'
        assert body['return_documents'] is False
        assert 'top_n' not in body
        assert requests[0].headers['Authorization'] == 'Bearer test-key'


def test_batches_apply_top_n_after_global_index_remapping() -> None:
    async def run() -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            bodies.append(payload)
            score = float(
                payload['documents'][0]['content'][0]['text'].split('-')[1]
            )
            return httpx.Response(
                200,
                json={
                    'results': [
                        {'index': index, 'relevance_score': score}
                        for index, _document in enumerate(payload['documents'])
                    ]
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            reranker = IonosReranker(
                _settings(batch_token_budget=3),
                _Budget(),  # type: ignore[arg-type]
                client,
            )
            results = await reranker.rerank(
                MultimodalContent(text='q'),
                [MultimodalContent(text=f'doc-{index}') for index in range(4)],
                content_images=[],
                top_n=2,
            )
        assert len(bodies) > 1
        assert len(results) == 2
        assert [item['index'] for item in results] == [3, 2]

    asyncio.run(run())


def test_empty_batch_does_not_call_http_or_encode_images() -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as client:
            reranker = IonosReranker(
                _settings(),
                _Budget(),
                client,  # type: ignore[arg-type]
            )
            assert (
                await reranker.rerank(
                    MultimodalContent(text='q'), [], content_images=[]
                )
                == []
            )

    asyncio.run(run())


def test_reranking_requires_credential_at_acquisition() -> None:
    with pytest.raises(ValueError, match='Reranking API key is required'):
        IonosReranker(
            RerankingSettings(),
            _Budget(),
            None,  # type: ignore[arg-type]
        )


def test_reranking_budget_repeats_query_image_cost() -> None:
    class Counter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [len(text) for text in texts]

    class Metadata:
        width = 32
        height = 32
        byte_count = 4

    class Cache:
        def measure(self, image: ContentImage) -> Metadata:
            return Metadata()

    budget = IonosRerankingBudget(Counter(), Cache())  # type: ignore[arg-type]
    query = MultimodalContent(text='q', image_positions=[0])
    docs = [MultimodalContent(text='a'), MultimodalContent(text='b')]
    images = [ContentImage(('image',), b'png')]
    with_query_image = budget.count_pairs(query, docs, content_images=images)
    without_query_image = budget.count_pairs(
        MultimodalContent(text='q'), docs, content_images=images
    )
    assert [
        with_image - without_image
        for with_image, without_image in zip(
            with_query_image, without_query_image, strict=True
        )
    ] == [1, 1]


def test_top_n_must_be_positive_for_nonempty_candidates() -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as client:
            reranker = IonosReranker(
                _settings(),
                _Budget(),
                client,  # type: ignore[arg-type]
            )
            with pytest.raises(ValueError, match='top_n must be positive'):
                await reranker.rerank(
                    MultimodalContent(text='q'),
                    [MultimodalContent(text='document')],
                    content_images=[],
                    top_n=0,
                )

    asyncio.run(run())


def test_reranking_rejects_duplicate_provider_indexes() -> None:
    async def run() -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'results': [
                        {'index': 0, 'relevance_score': 0.9},
                        {'index': 0, 'relevance_score': 0.8},
                    ]
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            reranker = IonosReranker(
                _settings(),
                _Budget(),
                client,  # type: ignore[arg-type]
            )
            with pytest.raises(
                ValueError, match='duplicate or invalid indexes'
            ):
                await reranker.rerank(
                    MultimodalContent(text='query'),
                    [
                        MultimodalContent(text='one'),
                        MultimodalContent(text='two'),
                    ],
                    content_images=[],
                )

    asyncio.run(run())


def test_reranking_rejects_nonfinite_provider_scores() -> None:
    async def run() -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=(b'{"results":[{"index":0,"relevance_score":NaN}]}'),
                headers={'Content-Type': 'application/json'},
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            reranker = IonosReranker(
                _settings(),
                _Budget(),
                client,  # type: ignore[arg-type]
            )
            with pytest.raises(ValueError, match='invalid relevance score'):
                await reranker.rerank(
                    MultimodalContent(text='query'),
                    [MultimodalContent(text='candidate')],
                    content_images=[],
                )

    asyncio.run(run())
