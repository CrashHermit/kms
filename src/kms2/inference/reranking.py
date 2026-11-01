"""Token-aware batching and HTTP reranking for Voyage AI."""

import asyncio
import math
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from kms2.config.reranking import RerankerProviderSettings
from kms2.core.reranking import RerankerClient, RerankResult
from kms2.core.windowing import TextTokenCounter

VOYAGE_RERANK_MODEL = 'rerank-3-lite'
VOYAGE_API_BASE = 'https://api.voyageai.com/v1'
MONGODB_API_BASE = 'https://ai.mongodb.com/v1'
MONGODB_API_KEY_PREFIX = 'al-'
VOYAGE_RERANK_URL = f'{VOYAGE_API_BASE}/rerank'
VOYAGE_MAX_QUERY_TOKENS = 8_000
VOYAGE_MAX_DOCUMENT_PAIR_TOKENS = 32_000
VOYAGE_MAX_TOTAL_TOKENS = 600_000
VOYAGE_MAX_DOCUMENTS = 1_000
MAX_RERANKER_BATCH_DOCUMENTS = VOYAGE_MAX_DOCUMENTS


class BatchingRerankerClient:
    """Batch candidates for one client; scores across batches may not compare.

    Each document's cost includes a repeated query count and backend overhead.
    A document exceeding the configured batch budget is sent alone, without
    truncation; the backend remains responsible for its individual limits.
    """

    def __init__(
        self,
        client: RerankerClient,
        token_counter: TextTokenCounter,
        *,
        token_budget: int,
        max_concurrency: int,
        overhead_tokens: int = 0,
        max_request_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._token_counter = token_counter
        self._token_budget = (
            min(token_budget, max_request_tokens)
            if max_request_tokens is not None
            else token_budget
        )
        self._overhead_tokens = overhead_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank every candidate, restoring original indexes after batching.

        Results from a single backend request retain provider order. Results
        merged from multiple requests sort by score and then original index;
        scores are not guaranteed comparable across requests.
        """
        if not documents:
            return []
        if top_n is not None and top_n < 1:
            raise ValueError('top_n must be positive when provided')

        query_count, *document_counts = self._token_counter.count_texts(
            [query, *documents]
        )
        pair_costs = [
            query_count + count + self._overhead_tokens
            for count in document_counts
        ]
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_cost = 0
        for index, pair_cost in enumerate(pair_costs):
            if pair_cost > self._token_budget:
                if batch:
                    batches.append(batch)
                    batch = []
                    batch_cost = 0
                batches.append([index])
                continue
            if batch and (
                batch_cost + pair_cost > self._token_budget
                or len(batch) >= MAX_RERANKER_BATCH_DOCUMENTS
            ):
                batches.append(batch)
                batch = []
                batch_cost = 0
            batch.append(index)
            batch_cost += pair_cost
        if batch:
            batches.append(batch)

        async def rerank_batch(
            indexes: list[int],
        ) -> tuple[list[int], list[RerankResult]]:
            async with self._semaphore:
                results = await self._client.rerank(
                    query, [documents[index] for index in indexes]
                )
            return indexes, results

        if len(batches) == 1:
            indexed_batches = [await rerank_batch(batches[0])]
        else:
            next_batch_index = 0
            indexed_batches_by_index: dict[
                int, tuple[list[int], list[RerankResult]]
            ] = {}

            async def rerank_batches() -> None:
                nonlocal next_batch_index
                while next_batch_index < len(batches):
                    batch_index = next_batch_index
                    next_batch_index += 1
                    indexed_batches_by_index[batch_index] = await rerank_batch(
                        batches[batch_index]
                    )

            tasks = [
                asyncio.create_task(rerank_batches())
                for _ in range(min(len(batches), self._max_concurrency))
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            indexed_batches = [
                indexed_batches_by_index[index] for index in range(len(batches))
            ]

        results: list[RerankResult] = []
        for indexes, batch_results in indexed_batches:
            results.extend(
                {
                    'index': indexes[result['index']],
                    'relevance_score': result['relevance_score'],
                }
                for result in batch_results
            )
        if len(batches) > 1:
            results.sort(
                key=lambda result: (
                    -result['relevance_score'],
                    result['index'],
                )
            )
        return results[:top_n] if top_n is not None else results


class VoyageRerankTokenCounter:
    """Count exact model tokens, including tokenizer-added special tokens."""

    def __init__(self, tokenizer_path: str | None = None) -> None:
        self._tokenizer = (
            Tokenizer.from_file(str(Path(tokenizer_path).expanduser()))
            if tokenizer_path is not None
            else Tokenizer.from_pretrained('voyageai/rerank-3-lite')
        )
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return exact model token counts in input order."""
        return [
            len(encoding)
            for encoding in self._tokenizer.encode_batch_fast(
                texts, add_special_tokens=True
            )
        ]


class VoyageRerankerClient:
    """Rerank with bounded concurrent Voyage requests and strict limits."""

    def __init__(
        self,
        settings: RerankerProviderSettings,
        token_counter: TextTokenCounter,
    ) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise ValueError('Voyage reranking requires a non-empty API key')
        self._settings = settings
        self._token_counter = token_counter
        self._api_key = settings.api_key.get_secret_value()
        self._url = (
            f'{MONGODB_API_BASE}/rerank'
            if self._api_key.startswith(MONGODB_API_KEY_PREFIX)
            else VOYAGE_RERANK_URL
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> VoyageRerankerClient:
        """Open the pooled async HTTP client for this resource lifetime."""
        self._client = httpx.AsyncClient(timeout=self._settings.timeout_seconds)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the pooled async HTTP client."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client if open."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return Voyage results in relevance order without truncating inputs."""
        if not documents:
            return []
        if len(documents) > VOYAGE_MAX_DOCUMENTS:
            raise ValueError('Voyage reranking accepts at most 1000 documents')
        if top_n is not None and top_n < 1:
            raise ValueError('top_n must be positive when provided')
        query_tokens, *document_tokens = self._token_counter.count_texts(
            [query, *documents]
        )
        if query_tokens > VOYAGE_MAX_QUERY_TOKENS:
            raise ValueError('Voyage rerank query exceeds 8000 tokens')
        if any(
            query_tokens + count > VOYAGE_MAX_DOCUMENT_PAIR_TOKENS
            for count in document_tokens
        ):
            raise ValueError(
                'Voyage rerank query and document exceed 32000 tokens'
            )
        if (
            query_tokens * len(documents) + sum(document_tokens)
            > VOYAGE_MAX_TOTAL_TOKENS
        ):
            raise ValueError('Voyage rerank request exceeds 600000 tokens')
        client = self._client
        if client is None:
            raise RuntimeError('Voyage reranker client is not open')
        async with self._semaphore:
            request_body: dict[str, object] = {
                'model': self._settings.model,
                'query': query,
                'documents': documents,
                'return_documents': False,
                'truncation': False,
            }
            if top_n is not None:
                request_body['top_k'] = top_n
            response = await client.post(
                self._url,
                headers={'Authorization': f'Bearer {self._api_key}'},
                json=request_body,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('Voyage rerank response must be an object')
        raw_results = payload.get('data')
        if not isinstance(raw_results, list):
            raise ValueError('Voyage rerank response data must be a list')
        expected_count = min(
            top_n if top_n is not None else len(documents), len(documents)
        )
        if len(raw_results) != expected_count:
            raise ValueError(
                'Voyage rerank response has an unexpected result count'
            )
        results: list[RerankResult] = []
        seen: set[int] = set()
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise ValueError('Voyage rerank result must be an object')
            index = raw_result.get('index')
            score = raw_result.get('relevance_score')
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen
            ):
                raise ValueError(
                    'Voyage rerank result has an invalid document index'
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(
                    'Voyage rerank result has an invalid relevance score'
                )
            seen.add(index)
            results.append({'index': index, 'relevance_score': score})
        return results
