"""Provider-neutral whole-input batching and concurrent embedding execution."""

import asyncio
from typing import cast

from kms.core.embedding import (
    Embedder,
    EmbeddingCapacity,
    EmbeddingEstimator,
    EmbeddingRequest,
    EmbeddingTransport,
)
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.media.images import prepare_image_uris


class BatchingEmbedder(Embedder):
    """Execute measured multimodal batches using injected model contracts."""

    def __init__(
        self,
        estimator: EmbeddingEstimator,
        capacity: EmbeddingCapacity,
        transport: EmbeddingTransport,
        *,
        batch_target_items: int,
        batch_target_tokens: int,
        max_concurrency: int,
    ) -> None:
        self._estimator = estimator
        self._capacity = capacity
        self._transport = transport
        self._batch_target_items = batch_target_items
        self._batch_target_tokens = batch_target_tokens
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Return one vector per material, regardless of batch completion order."""
        if not request.materials:
            return []
        costs = self._estimator.count_inputs(request)
        self._capacity.validate_items(request, costs)
        batches: list[tuple[int, int]] = []
        start = 0
        batch_cost = 0
        for index, cost in enumerate(costs):
            if not self._capacity.fits_batch(input_count=1, token_count=cost):
                raise InputBudgetExceeded(
                    'Embedding input exceeds the selected model request capacity'
                )
            batch_size = index - start
            if batch_size and (
                batch_size >= self._batch_target_items
                or batch_cost + cost > self._batch_target_tokens
                or not self._capacity.fits_batch(
                    input_count=batch_size + 1, token_count=batch_cost + cost
                )
            ):
                batches.append((start, index))
                start = index
                batch_cost = 0
            batch_cost += cost
        batches.append((start, len(request.materials)))
        image_uris = await prepare_image_uris(request.materials)
        results: list[list[float] | None] = [None] * len(request.materials)
        next_batch = iter(batches)

        async def worker() -> None:
            for start, end in next_batch:
                batch = request.model_copy(
                    update={'materials': request.materials[start:end]}
                )
                async with self._semaphore:
                    vectors = await self._transport.embed_batch(
                        batch, image_uris=image_uris
                    )
                results[start:end] = vectors

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._max_concurrency, len(batches)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return cast(list[list[float]], results)
