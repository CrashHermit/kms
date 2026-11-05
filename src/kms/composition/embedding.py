"""Composition for the pinned Voyage embedding capability."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from kms.config.settings import Settings
from kms.core.embedding import Embedder, EmbeddingModel
from kms.core.tokenization.images import ImageMetadataCache
from kms.inference.embedding.budget import (
    VOYAGE_EMBEDDING_DIMENSION,
    VoyageEmbeddingBudget,
)
from kms.inference.embedding.litellm import LiteLLMEmbedder
from kms.inference.tokenization.text import HuggingFaceTextTokenCounter


@dataclass(frozen=True, slots=True)
class EmbeddingResources:
    """Embedding implementation and its matching identity and budget."""

    embedder: Embedder
    model: EmbeddingModel
    budget: VoyageEmbeddingBudget


@asynccontextmanager
async def embedding_resources(
    settings: Settings,
) -> AsyncIterator[EmbeddingResources]:
    """Yield one fully matched Voyage adapter and budget for an operation."""
    provider_settings = settings.embedding
    counter = HuggingFaceTextTokenCounter(
        provider_settings.tokenizer_path, add_special_tokens=True
    )
    budget = VoyageEmbeddingBudget(
        text_counter=counter,
        image_metadata=ImageMetadataCache(),
    )
    embedder = LiteLLMEmbedder(provider_settings, budget)
    yield EmbeddingResources(
        embedder=embedder,
        model=EmbeddingModel(
            model=provider_settings.model,
            dimension=VOYAGE_EMBEDDING_DIMENSION,
        ),
        budget=budget,
    )
