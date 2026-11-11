"""LiteLLM batch transport with injected provider input conversion."""

import math
from collections.abc import Callable, Mapping
from typing import cast

import litellm

from kms.core.embedding import (
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingTransport,
)


class LiteLLMEmbeddingTransport(EmbeddingTransport):
    """Send one batch without owning model budgets or input conversion."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        build_inputs: Callable[
            [EmbeddingRequest, Mapping[tuple[str, ...], str]],
            list[dict[str, object]],
        ],
        request_options: Mapping[str, object],
        api_key: str,
        api_base: str | None,
        timeout_seconds: float,
    ) -> None:
        if not api_key:
            raise ValueError('Embedding API key is required')
        self._model = model
        self._build_inputs = build_inputs
        self._request_options = dict(request_options)
        self._api_key = api_key
        self._api_base = api_base
        self._timeout_seconds = timeout_seconds

    async def embed_batch(
        self,
        request: EmbeddingRequest,
        *,
        image_uris: Mapping[tuple[str, ...], str],
    ) -> list[list[float]]:
        """Restore provider-local indexes and validate the selected vector shape."""
        response = await litellm.aembedding(
            model=self._model.model,
            api_key=self._api_key,
            api_base=self._api_base,
            timeout=self._timeout_seconds,
            num_retries=0,
            input=self._build_inputs(request, image_uris),
            input_type=request.input_type,
            **self._request_options,
        )
        data = response['data'] if isinstance(response, dict) else response.data
        if len(data) != len(request.materials):
            raise ValueError(
                'Embedding provider returned an unexpected embedding count'
            )
        results: list[list[float] | None] = [None] * len(request.materials)
        seen_indexes: set[int] = set()
        for item in data:
            local_index = (
                item['index'] if isinstance(item, dict) else item.index
            )
            vector = (
                item['embedding'] if isinstance(item, dict) else item.embedding
            )
            if (
                type(local_index) is not int
                or not 0 <= local_index < len(request.materials)
                or local_index in seen_indexes
            ):
                raise ValueError(
                    'Embedding provider returned duplicate or invalid indexes'
                )
            if (
                not isinstance(vector, list)
                or len(vector) != self._model.dimension
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError(
                    'Embedding provider returned an invalid embedding vector'
                )
            seen_indexes.add(local_index)
            results[local_index] = vector
        return cast(list[list[float]], results)
