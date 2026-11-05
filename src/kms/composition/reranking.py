"""Composition for the pinned IONOS reranking capability."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from kms.config.settings import Settings
from kms.core.reranking import Reranker
from kms.core.tokenization.images import ImageMetadataCache
from kms.inference.reranking.budget import IonosRerankingBudget
from kms.inference.reranking.ionos import IonosReranker
from kms.inference.tokenization.text import HuggingFaceTextTokenCounter


@dataclass(frozen=True, slots=True)
class RerankingResources:
    """Reranking implementation and its matching token budget."""

    reranker: Reranker
    budget: IonosRerankingBudget


@asynccontextmanager
async def reranking_resources(
    settings: Settings,
) -> AsyncIterator[RerankingResources]:
    """Yield an IONOS adapter whose HTTP client closes on every exit path."""
    provider_settings = settings.reranker
    if (
        provider_settings.api_key is None
        or not provider_settings.api_key.get_secret_value()
    ):
        raise ValueError('Reranking API key is required')
    counter = HuggingFaceTextTokenCounter(
        provider_settings.tokenizer_path, add_special_tokens=False
    )
    budget = IonosRerankingBudget(counter, ImageMetadataCache())
    async with httpx.AsyncClient(
        timeout=provider_settings.timeout_seconds
    ) as client:
        yield RerankingResources(
            reranker=IonosReranker(provider_settings, budget, client),
            budget=budget,
        )
