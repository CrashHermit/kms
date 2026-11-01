"""Composition for selected reranking resources."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from kms2.config.settings import Settings
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import TextTokenCounter
from kms2.inference.reranking import (
    VOYAGE_MAX_TOTAL_TOKENS,
    BatchingRerankerClient,
    VoyageRerankerClient,
)
from kms2.local_models.runtime import LocalModelRuntime


@asynccontextmanager
async def reranker_resources(
    settings: Settings,
    local_models: LocalModelRuntime,
    token_counter: TextTokenCounter,
) -> AsyncIterator[RerankerClient]:
    """Yield the selected reranker behind shared request batching."""
    reranker_settings = settings.reranker
    if reranker_settings.backend == 'local':
        yield BatchingRerankerClient(
            local_models,
            token_counter,
            token_budget=reranker_settings.batch_token_budget,
            max_concurrency=reranker_settings.max_concurrency,
            overhead_tokens=(
                settings.local_models.reranker.safety_margin_tokens
            ),
        )
        return
    if reranker_settings.backend == 'voyage':
        async with VoyageRerankerClient(
            reranker_settings, token_counter
        ) as client:
            yield BatchingRerankerClient(
                client,
                token_counter,
                token_budget=reranker_settings.batch_token_budget,
                max_concurrency=reranker_settings.max_concurrency,
                max_request_tokens=VOYAGE_MAX_TOTAL_TOKENS,
            )
        return
    raise ValueError(
        f'Unsupported reranker backend: {reranker_settings.backend}'
    )
