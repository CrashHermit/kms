"""Behavioral tests for multimodal embedding execution and provider boundaries."""

import asyncio
import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from PIL import Image
from pydantic import SecretStr

from kms.config.embedding import EmbeddingSettings
from kms.core.embedding import (
    EmbeddingEstimator,
    EmbeddingModel,
    EmbeddingRequest,
)
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.inference.embedding.estimation import VoyageEmbeddingEstimator
from kms.inference.embedding.execution import BatchingEmbedder
from kms.inference.embedding.limits import (
    VOYAGE_EMBEDDING_DIMENSION,
    VoyageEmbeddingLimits,
)
from kms.inference.embedding.litellm import LiteLLMEmbeddingTransport
from kms.inference.embedding.voyage import voyage_inputs
from kms.media.metadata import ImageMetadata, ImageMetadataCache
from kms.media.models import ImageAsset, ImagePart, Material, TextPart


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _MetadataCache(ImageMetadataCache):
    def measure(self, _image: ImageAsset) -> ImageMetadata:
        return ImageMetadata(width=560, height=560, byte_count=100)


def _estimator() -> VoyageEmbeddingEstimator:
    return VoyageEmbeddingEstimator(_Counter(), _MetadataCache())


def _limits() -> VoyageEmbeddingLimits:
    return VoyageEmbeddingLimits(_MetadataCache())


def _settings(**overrides: object) -> EmbeddingSettings:
    return EmbeddingSettings(api_key=SecretStr('test-key'), **overrides)


def _embedder(
    settings: EmbeddingSettings,
    estimator: EmbeddingEstimator,
    limits: VoyageEmbeddingLimits,
) -> BatchingEmbedder:
    if settings.api_key is None:
        raise ValueError('Embedding API key is required')
    transport = LiteLLMEmbeddingTransport(
        EmbeddingModel(settings.model, VOYAGE_EMBEDDING_DIMENSION),
        build_inputs=voyage_inputs,
        request_options={
            'dimensions': VOYAGE_EMBEDDING_DIMENSION,
            'truncation': False,
        },
        api_key=settings.api_key.get_secret_value(),
        api_base=settings.api_base,
        timeout_seconds=settings.timeout_seconds,
    )
    return BatchingEmbedder(
        estimator,
        limits,
        transport,
        batch_target_items=settings.batch_target_items,
        batch_target_tokens=settings.batch_target_tokens,
        max_concurrency=settings.max_concurrency,
    )


def _request(
    materials: list[Material],
    *,
    input_type: Literal['query', 'document'] = 'document',
) -> EmbeddingRequest:
    return EmbeddingRequest(materials=materials, input_type=input_type)


def test_litellm_normalizes_native_images_without_losing_content_or_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    output = BytesIO()
    with Image.new('RGB', (128, 64), 'red') as image:
        image.save(output, format='PNG')
    image_bytes = output.getvalue()
    asset = ImageAsset(('red',), image_bytes)
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/v1/multimodalembeddings'
        body = json.loads(request.content)
        bodies.append(body)
        assert body['input_type'] == 'document'
        assert body['output_dimension'] == VOYAGE_EMBEDDING_DIMENSION
        assert body['truncation'] is False
        items = body['inputs']
        assert items[0] == {'content': [{'type': 'text', 'text': 'text-only'}]}
        image_part = items[1]['content'][0]
        assert image_part == {
            'type': 'image_base64',
            'image_base64': 'data:image/png;base64,'
            + base64.b64encode(image_bytes).decode('ascii'),
        }
        assert items[2]['content'] == [
            {'type': 'text', 'text': 'caption'},
            image_part,
            {'type': 'text', 'text': ' intervening text'},
            image_part,
        ]
        return httpx.Response(
            200,
            json={
                'object': 'list',
                'model': body['model'],
                'data': [
                    {
                        'index': index,
                        'embedding': [float(index)]
                        * VOYAGE_EMBEDDING_DIMENSION,
                    }
                    for index in (2, 0, 1)
                ],
                'usage': {'total_tokens': 12},
            },
        )

    async def run() -> None:
        actual_embedding = adapter.litellm.aembedding
        client = AsyncHTTPHandler(transport=httpx.MockTransport(handler))

        async def embed_with_client(**kwargs: object) -> object:
            return await actual_embedding(**kwargs, client=client)

        monkeypatch.setattr(adapter.litellm, 'aembedding', embed_with_client)
        try:
            embedder = _embedder(
                _settings(api_base='https://embedding.example/v1'),
                _estimator(),
                _limits(),
            )
            vectors = await embedder.embed(
                _request(
                    [
                        Material.from_text('text-only'),
                        Material.from_image(asset),
                        Material(
                            parts=(
                                TextPart(text='caption'),
                                ImagePart(asset=asset),
                                TextPart(text=' intervening text'),
                                ImagePart(asset=asset),
                            )
                        ),
                    ]
                )
            )
            assert vectors == [
                [float(index)] * VOYAGE_EMBEDDING_DIMENSION
                for index in range(3)
            ]
            assert len(bodies) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_estimation_is_independent_and_soft_singletons_are_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    calls: list[dict[str, object]] = []

    async def fake_embedding(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            'data': [
                {
                    'index': 0,
                    'embedding': [float(len(calls))]
                    * VOYAGE_EMBEDDING_DIMENSION,
                }
            ]
        }

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    request = _request(
        [Material.from_text('long text'), Material.from_text('longer')],
        input_type='query',
    )
    estimator = _estimator()
    assert estimator.count_inputs(request)[0] > 1
    embedder = _embedder(
        _settings(batch_target_tokens=1, max_concurrency=1),
        estimator,
        _limits(),
    )
    vectors = asyncio.run(embedder.embed(request))
    assert [len(call['input']) for call in calls] == [1, 1]  # type: ignore[arg-type]
    assert [vector[0] for vector in vectors] == [1.0, 2.0]


def test_empty_embedding_batch_needs_no_estimator_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    def fail(**kwargs: object) -> None:
        raise AssertionError('empty batch must not dispatch')

    monkeypatch.setattr(adapter.litellm, 'aembedding', fail)
    embedder = _embedder(_settings(), _estimator(), _limits())
    assert asyncio.run(embedder.embed(_request([]))) == []


def test_embedding_requires_credential_at_acquisition() -> None:
    with pytest.raises(ValueError, match='Embedding API key is required'):
        LiteLLMEmbeddingTransport(
            EmbeddingModel('test-model', 3),
            build_inputs=voyage_inputs,
            request_options={},
            api_key='',
            api_base=None,
            timeout_seconds=120,
        )


def test_estimator_counts_repeated_image_occurrences_without_capacity_rejection() -> (
    None
):
    image = ImageAsset(('shared',), b'image')
    request = _request([Material.from_image(image), Material.from_image(image)])
    counts = _estimator().count_inputs(request)
    assert counts[0] == counts[1]
    assert counts[0] > 560 + 256


def test_hard_over_limit_rejects_before_image_preparation_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    class LargeCounter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [32_001 for _ in texts]

    estimator = VoyageEmbeddingEstimator(LargeCounter(), _MetadataCache())
    request = _request([Material.from_text('text')])
    assert estimator.count_inputs(request) == [32_001 + 256]

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError('hard item limit must reject before preparation')

    monkeypatch.setattr(
        'kms.inference.embedding.execution.prepare_image_uris', fail
    )
    monkeypatch.setattr(adapter.litellm, 'aembedding', fail)
    embedder = _embedder(_settings(), estimator, _limits())
    with pytest.raises(ValueError, match='32,000 estimated tokens'):
        asyncio.run(embedder.embed(request))


def test_image_hard_limit_rejects_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    class OversizedCache(ImageMetadataCache):
        def measure(self, _image: ImageAsset) -> ImageMetadata:
            return ImageMetadata(width=4001, height=4000, byte_count=100)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError('image limit must reject before preparation')

    monkeypatch.setattr(
        'kms.inference.embedding.execution.prepare_image_uris', fail
    )
    monkeypatch.setattr(adapter.litellm, 'aembedding', fail)
    request = _request([Material.from_image(ImageAsset(('large',), b'image'))])
    embedder = _embedder(
        _settings(), _estimator(), VoyageEmbeddingLimits(OversizedCache())
    )
    with pytest.raises(ValueError, match='16,000,000 pixels'):
        asyncio.run(embedder.embed(request))


def test_embedding_rejects_duplicate_provider_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    async def fake_embedding(**_kwargs: object) -> dict[str, object]:
        return {
            'data': [
                {'index': 0, 'embedding': [1.0] * VOYAGE_EMBEDDING_DIMENSION},
                {'index': 0, 'embedding': [2.0] * VOYAGE_EMBEDDING_DIMENSION},
            ]
        }

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    embedder = _embedder(_settings(), _estimator(), _limits())
    with pytest.raises(ValueError, match='duplicate or invalid indexes'):
        asyncio.run(
            embedder.embed(
                _request(
                    [
                        Material.from_text('first'),
                        Material.from_text('second'),
                    ]
                )
            )
        )


def test_transport_validates_the_selected_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    vector = [0.25, -0.5, 1.0]

    async def fake_embedding(**_kwargs: object) -> dict[str, object]:
        return {'data': [{'index': 0, 'embedding': vector}]}

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    transport = LiteLLMEmbeddingTransport(
        EmbeddingModel('test-only/multimodal', 3),
        build_inputs=voyage_inputs,
        request_options={},
        api_key='test-key',
        api_base=None,
        timeout_seconds=120,
    )
    request = _request([Material.from_text('text')])
    assert asyncio.run(transport.embed_batch(request, image_uris={})) == [
        [0.25, -0.5, 1.0]
    ]
    vector = [1.0]
    with pytest.raises(ValueError, match='invalid embedding vector'):
        asyncio.run(transport.embed_batch(request, image_uris={}))


class _TextEstimator:
    def count_inputs(self, request: EmbeddingRequest) -> list[int]:
        return [len(material.text or '') for material in request.materials]


@dataclass(frozen=True)
class _Capacity:
    max_items: int = 3
    max_tokens: int = 5

    def validate_items(
        self, request: EmbeddingRequest, costs: list[int]
    ) -> None:
        if any(cost > 10 for cost in costs):
            raise InputBudgetExceeded('atomic input capacity')

    def fits_batch(self, *, input_count: int, token_count: int) -> bool:
        return input_count <= self.max_items and token_count <= self.max_tokens


def test_generic_execution_obeys_injected_limits_and_restores_completion_order() -> (
    None
):
    async def run() -> None:
        second_finished = asyncio.Event()
        batches: list[list[str]] = []
        finished: list[str] = []
        vectors = {
            'a': [1.0, 0.0, 0.0],
            'b': [0.0, 1.0, 0.0],
            'c': [0.0, 0.0, 1.0],
            'd': [1.0, 1.0, 0.0],
            'eeee': [1.0, 0.0, 1.0],
            'ff': [0.0, 1.0, 1.0],
        }

        class Transport:
            async def embed_batch(
                self,
                request: EmbeddingRequest,
                *,
                image_uris: Mapping[tuple[str, ...], str],
            ) -> list[list[float]]:
                texts = [material.text for material in request.materials]
                batches.append(texts)
                if texts[0] == 'a':
                    await second_finished.wait()
                else:
                    second_finished.set()
                finished.append(texts[0])
                return [vectors[text] for text in texts]

        embedder = BatchingEmbedder(
            _TextEstimator(),
            _Capacity(),
            Transport(),
            batch_target_items=99,
            batch_target_tokens=99,
            max_concurrency=2,
        )
        result = await asyncio.wait_for(
            embedder.embed(
                _request([Material.from_text(text) for text in vectors])
            ),
            timeout=2,
        )
        assert batches == [['a', 'b', 'c'], ['d', 'eeee'], ['ff']]
        assert finished[0] == 'd'
        assert result == list(vectors.values())

    asyncio.run(run())


def test_generic_request_capacity_rejects_singleton_before_loading_images() -> (
    None
):
    class Transport:
        async def embed_batch(
            self,
            request: EmbeddingRequest,
            *,
            image_uris: Mapping[tuple[str, ...], str],
        ) -> list[list[float]]:
            raise AssertionError('oversized input must not dispatch')

    embedder = BatchingEmbedder(
        _TextEstimator(),
        _Capacity(),
        Transport(),
        batch_target_items=99,
        batch_target_tokens=99,
        max_concurrency=1,
    )
    with pytest.raises(InputBudgetExceeded, match='model request capacity'):
        asyncio.run(
            embedder.embed(
                _request(
                    [
                        Material(
                            parts=(
                                TextPart(text='123456'),
                                ImagePart(
                                    asset=ImageAsset(
                                        ('unread',), b'not an image'
                                    )
                                ),
                            )
                        )
                    ]
                )
            )
        )


def test_generic_execution_joins_inflight_batches_after_transport_failure() -> (
    None
):
    async def run() -> None:
        second_started = asyncio.Event()
        second_released = asyncio.Event()

        class Transport:
            async def embed_batch(
                self,
                request: EmbeddingRequest,
                *,
                image_uris: Mapping[tuple[str, ...], str],
            ) -> list[list[float]]:
                if request.materials[0].text == 'first':
                    await second_started.wait()
                    raise RuntimeError('transport failed')
                second_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    second_released.set()

        embedder = BatchingEmbedder(
            _TextEstimator(),
            _Capacity(max_tokens=10),
            Transport(),
            batch_target_items=1,
            batch_target_tokens=99,
            max_concurrency=2,
        )
        with pytest.raises(RuntimeError, match='transport failed'):
            await asyncio.wait_for(
                embedder.embed(
                    _request(
                        [
                            Material.from_text('first'),
                            Material.from_text('second'),
                        ]
                    )
                ),
                timeout=2,
            )
        assert second_released.is_set()

    asyncio.run(run())
