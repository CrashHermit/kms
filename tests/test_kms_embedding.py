"""Behavioral tests for Voyage multimodal embedding."""

import asyncio

import pytest
from pydantic import SecretStr

from kms.config.embedding import EmbeddingSettings
from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.inference.embedding.budget import VoyageEmbeddingBudget
from kms.inference.embedding.litellm import LiteLLMEmbedder


class _Budget:
    def count_inputs(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: str,
    ) -> list[int]:
        return [len(material.text or '') + 1 for material in materials]


def _settings(**overrides: object) -> EmbeddingSettings:
    return EmbeddingSettings(api_key=SecretStr('test-key'), **overrides)


def test_embedding_preserves_provider_indexes_and_request_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    calls: list[dict[str, object]] = []

    async def fake_encode(images: list[ContentImage]) -> list[str]:
        return ['data:image/png;base64,AA==' for _ in images]

    async def fake_embedding(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            'data': [
                {'index': 1, 'embedding': [2.0] * 1024},
                {'index': 0, 'embedding': [1.0] * 1024},
            ]
        }

    monkeypatch.setattr(adapter, 'encode_content_images', fake_encode)
    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    embedder = LiteLLMEmbedder(_settings(), _Budget())  # type: ignore[arg-type]
    materials = [
        MultimodalContent(text='one'),
        MultimodalContent(text='two', image_positions=[0]),
    ]
    vectors = asyncio.run(
        embedder.embed(
            materials,
            content_images=[ContentImage(('image',), b'image')],
            input_type='document',
        )
    )
    assert [vector[0] for vector in vectors] == [1.0, 2.0]
    assert calls[0]['model'] == 'voyage/voyage-multimodal-3.5'
    assert calls[0]['input_type'] == 'document'
    assert calls[0]['dimensions'] == 1024
    assert calls[0]['truncation'] is False
    assert calls[0]['num_retries'] == 0
    assert calls[0]['input'][1]['content'][1]['type'] == 'image_base64'  # type: ignore[index]


def test_soft_budget_accepts_over_target_items_as_singletons(
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
                    'embedding': [float(len(calls))] * 1024,
                }
            ]
        }

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    embedder = LiteLLMEmbedder(
        _settings(batch_token_budget=1, max_concurrency=1),
        _Budget(),
    )  # type: ignore[arg-type]
    vectors = asyncio.run(
        embedder.embed(
            [
                MultimodalContent(text='oversized one'),
                MultimodalContent(text='oversized two'),
            ],
            content_images=[],
            input_type='query',
        )
    )
    assert [len(call['input']) for call in calls] == [1, 1]  # type: ignore[arg-type]
    assert [vector[0] for vector in vectors] == [1.0, 2.0]


def test_empty_embedding_batch_needs_no_tokenizer_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    def fail(**kwargs: object) -> None:
        raise AssertionError('empty batch must not dispatch')

    monkeypatch.setattr(adapter.litellm, 'aembedding', fail)
    embedder = LiteLLMEmbedder(_settings(), _Budget())  # type: ignore[arg-type]
    assert (
        asyncio.run(embedder.embed([], content_images=[], input_type='query'))
        == []
    )


def test_embedding_requires_credential_at_acquisition() -> None:
    with pytest.raises(ValueError, match='Embedding API key is required'):
        LiteLLMEmbedder(EmbeddingSettings(), _Budget())  # type: ignore[arg-type]


def test_voyage_budget_counts_repeated_image_occurrences() -> None:
    class Counter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [len(text) for text in texts]

    class Metadata:
        width = 560
        height = 560
        byte_count = 100

    class Cache:
        def measure(self, image: ContentImage) -> Metadata:
            return Metadata()

    budget = VoyageEmbeddingBudget(Counter(), Cache())  # type: ignore[arg-type]
    image = ContentImage(('shared',), b'image')
    materials = [
        MultimodalContent(image_positions=[0]),
        MultimodalContent(image_positions=[0]),
    ]
    counts = budget.count_inputs(
        materials,
        content_images=[image],
        input_type='document',
    )
    assert counts[0] == counts[1]
    assert counts[0] > 560 + 256


def test_embedding_rejects_duplicate_provider_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    async def fake_embedding(**_kwargs: object) -> dict[str, object]:
        return {
            'data': [
                {'index': 0, 'embedding': [1.0] * 1024},
                {'index': 0, 'embedding': [2.0] * 1024},
            ]
        }

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    embedder = LiteLLMEmbedder(_settings(), _Budget())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='duplicate or invalid indexes'):
        asyncio.run(
            embedder.embed(
                [
                    MultimodalContent(text='first'),
                    MultimodalContent(text='second'),
                ],
                content_images=[],
                input_type='document',
            )
        )


def test_embedding_rejects_wrong_vector_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kms.inference.embedding.litellm as adapter

    async def fake_embedding(**_kwargs: object) -> dict[str, object]:
        return {'data': [{'index': 0, 'embedding': [1.0]}]}

    monkeypatch.setattr(adapter.litellm, 'aembedding', fake_embedding)
    embedder = LiteLLMEmbedder(_settings(), _Budget())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='invalid embedding vector'):
        asyncio.run(
            embedder.embed(
                [MultimodalContent(text='text')],
                content_images=[],
                input_type='document',
            )
        )
