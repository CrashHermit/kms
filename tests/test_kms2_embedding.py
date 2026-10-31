"""Behavioral tests for selected KMS2 embedding clients."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from kms2.composition.embedding import (
    build_embedding_client,
    embedding_dimension,
)
from kms2.config.embedding import EmbeddingProviderSettings
from kms2.config.settings import Settings
from kms2.inference import embedding


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _settings(**overrides: object) -> EmbeddingProviderSettings:
    return EmbeddingProviderSettings(api_key=SecretStr('test-key'), **overrides)


def _response(vectors: list[list[float]]) -> dict[str, object]:
    return {
        'data': [
            {'index': index, 'embedding': vector}
            for index, vector in enumerate(vectors)
        ]
    }


def test_batches_and_provider_indices_restore_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def aembedding(**kwargs: object) -> dict[str, object]:
        texts = kwargs['input']
        assert isinstance(texts, list)
        calls.append(texts)
        vectors = [[float(ord(text[0]))] * 1024 for text in texts]
        return {
            'data': [
                {'index': index, 'embedding': vector}
                for index, vector in reversed(list(enumerate(vectors)))
            ]
        }

    monkeypatch.setattr(embedding.litellm, 'aembedding', aembedding)
    client = embedding.OpenRouterEmbeddingClient(
        _settings(batch_size=2, batch_token_budget=10), _Counter()
    )
    vectors = asyncio.run(client.embed(['a', 'bb', 'ccc', 'd']))
    assert calls == [['a', 'bb'], ['ccc', 'd']]
    assert [vector[0] for vector in vectors] == [97.0, 98.0, 99.0, 100.0]


def test_concurrency_is_bounded_across_simultaneous_embed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    release = asyncio.Event()
    saturated = asyncio.Event()

    async def aembedding(**kwargs: object) -> dict[str, object]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            saturated.set()
        await release.wait()
        active -= 1
        return _response([[1.0] * 1024 for _ in kwargs['input']])

    monkeypatch.setattr(embedding.litellm, 'aembedding', aembedding)
    client = embedding.OpenRouterEmbeddingClient(
        _settings(max_concurrency=2, batch_size=1), _Counter()
    )

    async def run_calls() -> list[list[list[float]]]:
        tasks = [
            asyncio.create_task(client.embed(['x', 'y'])) for _ in range(3)
        ]
        await asyncio.wait_for(saturated.wait(), timeout=1)
        assert maximum == 2
        release.set()
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_calls())
    assert len(results) == 3
    assert maximum == 2


def test_rejects_indivisible_overbudget_item() -> None:
    client = embedding.OpenRouterEmbeddingClient(
        _settings(batch_token_budget=1), _Counter()
    )
    with pytest.raises(ValueError, match='cannot fit as a singleton'):
        asyncio.run(client.embed(['too-long']))


def test_per_input_limit_accepts_boundary_and_rejects_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = [1.0] * 1024

    async def aembedding(**kwargs: object) -> dict[str, object]:
        return _response([vector])

    monkeypatch.setattr(embedding.litellm, 'aembedding', aembedding)
    settings = _settings(batch_token_budget=40_000)
    client = embedding.OpenRouterEmbeddingClient(settings, _Counter())
    assert asyncio.run(client.embed(['x' * 32_000])) == [vector]
    with pytest.raises(ValueError):
        asyncio.run(client.embed(['x' * 32_001]))


@pytest.mark.parametrize(
    ('response', 'message'),
    [
        ({'data': []}, 'wrong item count'),
        (
            {'data': [{'index': 1, 'embedding': [0.0] * 1024}]},
            'invalid item index',
        ),
        (
            {'data': [{'index': 0, 'embedding': [0.0]}]},
            'wrong vector dimension',
        ),
    ],
)
def test_rejects_invalid_provider_results(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, object],
    message: str,
) -> None:
    async def aembedding(**kwargs: object) -> dict[str, object]:
        return response

    monkeypatch.setattr(embedding.litellm, 'aembedding', aembedding)
    client = embedding.OpenRouterEmbeddingClient(_settings(), _Counter())
    with pytest.raises(ValueError, match=message):
        asyncio.run(client.embed(['x']))


def test_local_selection_does_not_start_gpu_runtime() -> None:
    runtime = SimpleNamespace()
    settings = Settings(embedding={'backend': 'local'})
    client = build_embedding_client(settings, runtime, _Counter())
    assert client is runtime
    assert (
        embedding_dimension(settings)
        == settings.local_models.embedding.model.dimension
    )


def test_hosted_selection_dimension_and_missing_key_boundary() -> None:
    settings = Settings(_env_file=None, embedding={'api_key': None})
    assert embedding_dimension(settings) == 1024
    with pytest.raises(ValueError, match='API key'):
        build_embedding_client(settings, SimpleNamespace(), _Counter())


def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match='non-empty API key'):
        embedding.OpenRouterEmbeddingClient(
            EmbeddingProviderSettings(api_key=SecretStr('')), _Counter()
        )
