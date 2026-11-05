"""LiteLLM transport for Voyage's native multimodal embedding API."""

import asyncio
import math
from typing import Literal, cast

import litellm

from kms.config.embedding import EmbeddingSettings
from kms.core.content import ContentImage
from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.inference.embedding.budget import (
    VOYAGE_EMBEDDING_DIMENSION,
    VOYAGE_MAX_INPUTS,
    VOYAGE_MAX_TOTAL_TOKENS,
    VoyageEmbeddingBudget,
)
from kms.service.content import encode_content_images


class LiteLLMEmbedder(Embedder):
    """Embed ordered multimodal materials with bounded concurrent batches."""

    def __init__(
        self, settings: EmbeddingSettings, budget: VoyageEmbeddingBudget
    ) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise ValueError('Embedding API key is required')
        self._settings = settings
        self._budget = budget
        self._api_key = settings.api_key.get_secret_value()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def embed(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: Literal['query', 'document'],
    ) -> list[list[float]]:
        if not materials:
            return []
        costs = self._budget.count_inputs(
            materials, content_images=content_images, input_type=input_type
        )
        batches: list[list[tuple[int, MultimodalContent]]] = []
        batch: list[tuple[int, MultimodalContent]] = []
        batch_cost = 0
        batch_size = min(self._settings.batch_size, VOYAGE_MAX_INPUTS)
        for item, cost in zip(enumerate(materials), costs, strict=True):
            if batch and (
                batch_cost + cost > self._settings.batch_token_budget
                or len(batch) >= batch_size
            ):
                batches.append(batch)
                batch = []
                batch_cost = 0
            batch.append(item)
            batch_cost += cost
        if batch:
            batches.append(batch)
        for batch in batches:
            if (
                sum(costs[index] for index, _ in batch)
                > VOYAGE_MAX_TOTAL_TOKENS
            ):
                raise InputBudgetExceeded(
                    'Voyage embedding request exceeds 320,000 tokens'
                )
        referenced_positions = sorted(
            {
                position
                for material in materials
                for position in material.image_positions
            }
        )
        uris = await encode_content_images(
            [content_images[position] for position in referenced_positions]
        )
        image_uri_by_position = dict(
            zip(referenced_positions, uris, strict=True)
        )
        results: list[list[float] | None] = [None] * len(materials)
        next_batch = iter(batches)

        async def worker() -> None:
            for batch in next_batch:
                request_items = []
                for _, material in batch:
                    parts: list[dict[str, object]] = []
                    if material.text:
                        parts.append({'type': 'text', 'text': material.text})
                    parts.extend(
                        {
                            'type': 'image_base64',
                            'image_base64': image_uri_by_position[position],
                        }
                        for position in material.image_positions
                    )
                    request_items.append({'content': parts})
                async with self._semaphore:
                    response = await litellm.aembedding(
                        model=self._settings.model,
                        input=request_items,
                        api_key=self._api_key,
                        input_type=input_type,
                        dimensions=VOYAGE_EMBEDDING_DIMENSION,
                        truncation=False,
                        num_retries=0,
                        timeout=self._settings.timeout_seconds,
                    )
                data = (
                    response['data']
                    if isinstance(response, dict)
                    else response.data
                )
                if len(data) != len(batch):
                    raise ValueError(
                        'Voyage returned an unexpected embedding count'
                    )
                seen_indexes: set[int] = set()
                for item in data:
                    local_index = (
                        item['index'] if isinstance(item, dict) else item.index
                    )
                    vector = (
                        item['embedding']
                        if isinstance(item, dict)
                        else item.embedding
                    )
                    if (
                        type(local_index) is not int
                        or not 0 <= local_index < len(batch)
                        or local_index in seen_indexes
                    ):
                        raise ValueError(
                            'Voyage returned duplicate or invalid indexes'
                        )
                    if (
                        not isinstance(vector, list)
                        or len(vector) != VOYAGE_EMBEDDING_DIMENSION
                        or any(
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                            for value in vector
                        )
                    ):
                        raise ValueError(
                            'Voyage returned an invalid embedding vector'
                        )
                    seen_indexes.add(local_index)
                    original_index = batch[local_index][0]
                    results[original_index] = vector

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._settings.max_concurrency, len(batches)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return cast(list[list[float]], results)
