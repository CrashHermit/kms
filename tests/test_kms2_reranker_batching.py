"""Behavioral tests for the shared reranker batching boundary."""

import asyncio
from collections.abc import Callable

import pytest
from pydantic import SecretStr

from kms2.composition import reranking as reranking_composition
from kms2.config.reranking import RerankerProviderSettings
from kms2.config.settings import Settings
from kms2.core.reranking import RerankResult
from kms2.inference.reranking import (
    MAX_RERANKER_BATCH_DOCUMENTS,
    BatchingRerankerClient,
)


class _Counter:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls.append(texts)
        return [len(text) for text in texts]


class _Client:
    def __init__(
        self,
        score: Callable[[str], float] | None = None,
        delay: float = 0,
    ) -> None:
        self.score = score or (lambda _document: 0.0)
        self.delay = delay
        self.requests: list[list[str]] = []
        self.active = 0
        self.peak = 0

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        self.requests.append(documents)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return [
                {'index': index, 'relevance_score': self.score(document)}
                for index, document in enumerate(documents)
            ]
        finally:
            self.active -= 1


def test_batches_full_pairs_and_applies_global_top_n_with_original_indexes() -> (
    None
):
    client = _Client(score=lambda document: float(document))
    counter = _Counter()
    reranker = BatchingRerankerClient(
        client,
        counter,
        token_budget=10,
        max_concurrency=2,
        overhead_tokens=1,
    )

    results = asyncio.run(
        reranker.rerank('qq', ['1', '222', '333', '444'], top_n=2)
    )

    # Each pair costs query (2) + document (1 or 3) + overhead (1).
    assert client.requests == [['1', '222'], ['333'], ['444']]
    assert counter.calls == [['qq', '1', '222', '333', '444']]
    assert results == [
        {'index': 3, 'relevance_score': 444.0},
        {'index': 2, 'relevance_score': 333.0},
    ]


def test_one_batch_retains_provider_order() -> None:
    class ProviderOrderClient(_Client):
        async def rerank(
            self,
            query: str,
            documents: list[str],
            top_n: int | None = None,
        ) -> list[RerankResult]:
            self.requests.append(documents)
            return [
                {'index': 1, 'relevance_score': 0.1},
                {'index': 0, 'relevance_score': 0.9},
            ]

    results = asyncio.run(
        BatchingRerankerClient(
            ProviderOrderClient(),
            _Counter(),
            token_budget=100,
            max_concurrency=2,
        ).rerank('q', ['a', 'b'])
    )

    assert [result['index'] for result in results] == [1, 0]


def test_batches_cap_document_count_at_voyage_limit() -> None:
    client = _Client()
    results = asyncio.run(
        BatchingRerankerClient(
            client,
            _Counter(),
            token_budget=10_000,
            max_concurrency=1,
        ).rerank('q', ['d'] * (MAX_RERANKER_BATCH_DOCUMENTS + 1))
    )

    assert [len(request) for request in client.requests] == [1000, 1]
    assert len(results) == MAX_RERANKER_BATCH_DOCUMENTS + 1
    assert results[-1]['index'] == MAX_RERANKER_BATCH_DOCUMENTS


def test_parallel_batches_obey_configured_concurrency() -> None:
    client = _Client(delay=0.01)
    asyncio.run(
        BatchingRerankerClient(
            client, _Counter(), token_budget=1, max_concurrency=2
        ).rerank('q', ['a'] * 7)
    )

    assert client.peak == 2


def test_concurrency_limit_is_shared_across_rerank_calls() -> None:
    async def run() -> None:
        client = _Client(delay=0.01)
        reranker = BatchingRerankerClient(
            client, _Counter(), token_budget=1, max_concurrency=2
        )
        await asyncio.gather(
            reranker.rerank('q', ['a'] * 3),
            reranker.rerank('q', ['b'] * 3),
        )
        assert client.peak == 2

    asyncio.run(run())


def test_cancelled_multi_batch_request_waits_for_siblings() -> None:
    async def run() -> None:
        class BlockingClient(_Client):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = 0

            async def rerank(
                self,
                query: str,
                documents: list[str],
                top_n: int | None = None,
            ) -> list[RerankResult]:
                self.active += 1
                if self.active == 2:
                    self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.active -= 1
                    self.cancelled += 1

        client = BlockingClient()
        reranker = BatchingRerankerClient(
            client, _Counter(), token_budget=1, max_concurrency=2
        )
        request = asyncio.create_task(reranker.rerank('q', ['a'] * 3))
        await client.started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert client.active == 0
        assert client.cancelled == 2

    asyncio.run(run())


def test_oversized_pair_is_sent_alone_without_truncation() -> None:
    client = _Client(score=lambda _document: 0.7)
    document = 'whole pair exceeds configured budget'
    results = asyncio.run(
        BatchingRerankerClient(
            client, _Counter(), token_budget=2, max_concurrency=1
        ).rerank('q', [document])
    )

    assert client.requests == [[document]]
    assert results == [{'index': 0, 'relevance_score': 0.7}]


def test_provider_errors_propagate_from_batched_requests() -> None:
    class FailingClient(_Client):
        async def rerank(
            self,
            query: str,
            documents: list[str],
            top_n: int | None = None,
        ) -> list[RerankResult]:
            raise RuntimeError('provider failure')

    async def run() -> None:
        reranker = BatchingRerankerClient(
            FailingClient(), _Counter(), token_budget=1, max_concurrency=2
        )
        with pytest.raises(RuntimeError, match='provider failure'):
            await reranker.rerank('q', ['a', 'b'])

    asyncio.run(run())


def test_equal_scores_across_batches_break_ties_by_original_index() -> None:
    client = _Client(score=lambda _document: 0.5)
    results = asyncio.run(
        BatchingRerankerClient(
            client, _Counter(), token_budget=1, max_concurrency=3
        ).rerank('q', ['a', 'b', 'c'])
    )

    assert [result['index'] for result in results] == [0, 1, 2]


def test_voyage_composition_caps_batches_at_provider_total_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResourceClient(_Client):
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

    raw_client = ResourceClient()
    monkeypatch.setattr(
        reranking_composition,
        'VoyageRerankerClient',
        lambda *args: raw_client,
    )

    class Counter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [1_000, *([29_000] * (len(texts) - 1))]

    settings = Settings(
        reranker=RerankerProviderSettings(
            backend='voyage',
            api_key=SecretStr('voyage-key'),
            batch_token_budget=1_000_000,
        )
    )
    documents = [str(index) for index in range(21)]

    async def rerank() -> None:
        async with reranking_composition.reranker_resources(
            settings, _Client(), Counter()
        ) as client:
            await client.rerank('query', documents)

    asyncio.run(rerank())

    assert raw_client.requests == [documents[:20], documents[20:]]
