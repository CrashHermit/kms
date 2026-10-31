"""Behavioral tests for hosted Voyage reranking contracts."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from pydantic import SecretStr

from kms2.config.reranking import RerankerProviderSettings
from kms2.inference.reranking import VoyageRerankerClient


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _client(max_concurrency: int = 2) -> VoyageRerankerClient:
    return VoyageRerankerClient(
        RerankerProviderSettings(
            api_key=SecretStr('voyage-key'), max_concurrency=max_concurrency
        ),
        _Counter(),
    )


@asynccontextmanager
async def _client_with_transport(
    transport: httpx.AsyncBaseTransport,
    *,
    max_concurrency: int = 2,
) -> AsyncIterator[VoyageRerankerClient]:
    client = _client(max_concurrency)
    client._client = httpx.AsyncClient(transport=transport)
    try:
        yield client
    finally:
        await client.close()


def _result(index: int = 0) -> dict[str, object]:
    return {'index': index, 'relevance_score': 0.9}


def test_rerank_preserves_provider_rank_and_original_indexes() -> None:
    asyncio.run(_rerank_preserves_provider_rank_and_original_indexes())


async def _rerank_preserves_provider_rank_and_original_indexes() -> None:
    async with _client_with_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={'data': [_result(2), _result(0)]},
            )
        )
    ) as client:
        results = await client.rerank('q', ['a', 'b', 'c'], top_n=2)
    assert [result['index'] for result in results] == [2, 0]


def test_rerank_limits_parallel_requests_across_calls() -> None:
    asyncio.run(_rerank_limits_parallel_requests_across_calls())


async def _rerank_limits_parallel_requests_across_calls() -> None:
    active = 0
    peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={'data': [_result()]})

    async with _client_with_transport(
        httpx.MockTransport(respond), max_concurrency=2
    ) as client:
        await asyncio.gather(*(client.rerank('q', ['d']) for _ in range(5)))
    assert peak == 2


def test_cancelled_request_releases_concurrency_slot() -> None:
    asyncio.run(_cancelled_request_releases_concurrency_slot())


async def _cancelled_request_releases_concurrency_slot() -> None:
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        entered.set()
        await finish.wait()
        return httpx.Response(200, json={'data': [_result()]})

    async with _client_with_transport(
        httpx.MockTransport(respond), max_concurrency=1
    ) as client:
        cancelled = asyncio.create_task(client.rerank('q', ['d']))
        await entered.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        finish.set()
        assert await client.rerank('q', ['d']) == [_result()]


def test_rerank_rejects_token_limit_excess_and_accepts_boundary() -> None:
    asyncio.run(_rerank_rejects_token_limit_excess_and_accepts_boundary())


async def _rerank_rejects_token_limit_excess_and_accepts_boundary() -> None:
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda request: (
            requests.append(request)
            or httpx.Response(200, json={'data': [_result()]})
        )
    )
    async with _client_with_transport(transport) as client:
        await client.rerank('q', ['x' * 31_999])
        with pytest.raises(ValueError, match='query and document'):
            await client.rerank('q', ['x' * 32_000])
    assert len(requests) == 1


def test_rerank_propagates_provider_http_errors() -> None:
    asyncio.run(_rerank_propagates_provider_http_errors())


async def _rerank_propagates_provider_http_errors() -> None:
    async with _client_with_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(429, text='rate limited')
        )
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.rerank('q', ['d'])


@pytest.mark.parametrize(
    ('data', 'message'),
    [
        ([], 'result count'),
        ([_result(0), _result(0)], 'document index'),
        ([_result(-1), _result(0)], 'document index'),
        ([_result(0), _result(2)], 'document index'),
        ([_result(True), _result(0)], 'document index'),
        (
            [
                {'index': 0, 'relevance_score': float('nan')},
                _result(1),
            ],
            'relevance score',
        ),
    ],
)
def test_rejects_results_that_could_misattribute_candidates(
    data: list[dict[str, object]], message: str
) -> None:
    async def run() -> None:
        async with _client_with_transport(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=json.dumps({'data': data})
                )
            )
        ) as client:
            with pytest.raises(ValueError, match=message):
                await client.rerank('query', ['first', 'second'])

    asyncio.run(run())


@pytest.mark.parametrize(
    ('query', 'documents', 'message'),
    [
        ('q' * 8001, ['document'], 'query exceeds'),
        ('q', ['document'] * 1001, 'at most 1000'),
        ('q' * 8000, ['d' * 22000] * 21, '600000'),
    ],
)
def test_rejects_oversized_requests_before_network_io(
    query: str, documents: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        asyncio.run(_client().rerank(query, documents))
