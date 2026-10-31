"""Composition for selected reranking resources."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from kms2.config.settings import Settings
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import TextTokenCounter
from kms2.inference.reranking import VoyageRerankerClient
from kms2.local_models.runtime import LocalModelRuntime


@asynccontextmanager
async def reranker_resources(
    settings: Settings,
    local_models: LocalModelRuntime,
    token_counter: TextTokenCounter,
) -> AsyncIterator[RerankerClient]:
    """Yield the selected reranker while owning only hosted client lifetime."""
    if settings.reranker.backend == 'local':
        yield local_models
        return
    if settings.reranker.backend == 'voyage':
        async with VoyageRerankerClient(
            settings.reranker, token_counter
        ) as client:
            yield client
        return
    raise ValueError(
        f'Unsupported reranker backend: {settings.reranker.backend}'
    )


def reranker_overhead_tokens(settings: Settings) -> int:
    """Return the reserved input margin for the selected reranker."""
    if settings.reranker.backend == 'local':
        return settings.local_models.reranker.safety_margin_tokens
    if settings.reranker.backend == 'voyage':
        return 0
    raise ValueError(
        f'Unsupported reranker backend: {settings.reranker.backend}'
    )
