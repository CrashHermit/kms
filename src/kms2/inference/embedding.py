"""LiteLLM-backed OpenRouter embedding client."""

import asyncio
from pathlib import Path
from typing import cast

import litellm
from tokenizers import Tokenizer

from kms2.config.embedding import EmbeddingProviderSettings
from kms2.core.windowing import TextTokenCounter, pack_items

VOYAGE_EMBEDDING_MAX_INPUT_TOKENS = 32_000


class VoyageEmbeddingTokenCounter:
    """Count Voyage 4 Large input tokens including model special tokens."""

    def __init__(self, tokenizer_path: str | None = None) -> None:
        self._tokenizer = (
            Tokenizer.from_file(str(Path(tokenizer_path).expanduser()))
            if tokenizer_path is not None
            else Tokenizer.from_pretrained('voyageai/voyage-4-large')
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


class OpenRouterEmbeddingClient:
    """Embed via LiteLLM with bounded batches and ordered result validation."""

    def __init__(
        self,
        settings: EmbeddingProviderSettings,
        token_counter: TextTokenCounter,
    ) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise ValueError(
                'OpenRouter embedding requires a non-empty API key'
            )
        self._settings = settings
        self._token_counter = token_counter
        self._api_key = settings.api_key.get_secret_value()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one 1024-dimensional vector per text in original order."""
        if not texts:
            return []
        counts = self._token_counter.count_texts(texts)
        if any(count > VOYAGE_EMBEDDING_MAX_INPUT_TOKENS for count in counts):
            raise ValueError(
                'Embedding input exceeds Voyage 4 Large 32000-token limit'
            )
        batches = pack_items(
            texts,
            token_counts=counts,
            token_budget=self._settings.batch_token_budget,
            max_items=self._settings.batch_size,
        )
        results: list[list[list[float]] | None] = [None] * len(batches)
        next_batch = iter(enumerate(batches))

        async def worker() -> None:
            for batch_index, batch in next_batch:
                async with self._semaphore:
                    response = await litellm.aembedding(
                        model=f'openrouter/{self._settings.model}',
                        input=batch,
                        api_key=self._api_key,
                        timeout=self._settings.timeout_seconds,
                    )
                data = (
                    response['data']
                    if isinstance(response, dict)
                    else response.data
                )
                if len(data) != len(batch):
                    raise ValueError(
                        'Embedding provider returned wrong item count'
                    )
                ordered: list[list[float] | None] = [None] * len(batch)
                for item in data:
                    index = (
                        item['index'] if isinstance(item, dict) else item.index
                    )
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or not 0 <= index < len(batch)
                    ):
                        raise ValueError(
                            'Embedding provider returned invalid item index'
                        )
                    vector = (
                        item['embedding']
                        if isinstance(item, dict)
                        else item.embedding
                    )
                    if ordered[index] is not None:
                        raise ValueError(
                            'Embedding provider returned duplicate item index'
                        )
                    if len(vector) != 1024:
                        raise ValueError(
                            'Embedding provider returned wrong vector dimension'
                        )
                    ordered[index] = cast(list[float], vector)
                if any(vector is None for vector in ordered):
                    raise ValueError('Embedding provider omitted an item index')
                results[batch_index] = cast(list[list[float]], ordered)

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
        return [
            vector
            for result in results
            for vector in cast(list[list[float]], result)
        ]
