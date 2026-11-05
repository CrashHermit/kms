"""Behavioral tests for ordered IONOS candidate batching."""

import asyncio
import json

import httpx

from kms.config.reranking import RerankingSettings
from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.inference.reranking.ionos import IonosReranker


class _Budget:
    image_metadata = None

    def count_pairs(
        self,
        query: MultimodalContent,
        documents: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
    ) -> list[int]:
        return [2 for _ in documents]


def test_multibatch_scores_use_original_indexes_for_global_top_n() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            documents = json.loads(request.content)['documents']
            score = float(documents[0]['content'][0]['text'].split('-')[1])
            return httpx.Response(
                200,
                json={
                    'results': [
                        {'index': index, 'relevance_score': score}
                        for index, _document in enumerate(documents)
                    ]
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            reranker = IonosReranker(
                RerankingSettings(
                    api_key='key', batch_token_budget=2, max_concurrency=2
                ),
                _Budget(),
                client,
            )
            results = await reranker.rerank(
                MultimodalContent(text='query'),
                [MultimodalContent(text=f'doc-{index}') for index in range(4)],
                content_images=[],
                top_n=2,
            )
        assert results == [
            {'index': 3, 'relevance_score': 3.0},
            {'index': 2, 'relevance_score': 2.0},
        ]

    asyncio.run(run())


def test_one_request_retains_provider_rank_order() -> None:
    async def run() -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    'results': [
                        {'index': 1, 'relevance_score': 0.1},
                        {'index': 0, 'relevance_score': 0.9},
                    ]
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            reranker = IonosReranker(
                RerankingSettings(api_key='key'), _Budget(), client
            )
            results = await reranker.rerank(
                MultimodalContent(text='q'),
                [MultimodalContent(text='a'), MultimodalContent(text='b')],
                content_images=[],
            )
        assert [result['index'] for result in results] == [1, 0]

    asyncio.run(run())
