"""Compose the selected embedding model, rules, transport and execution policy."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from kms.config.settings import Settings
from kms.core.embedding import (
    Embedder,
    EmbeddingCapacity,
    EmbeddingEstimator,
    EmbeddingModel,
)
from kms.inference.embedding.estimation import VoyageEmbeddingEstimator
from kms.inference.embedding.execution import BatchingEmbedder
from kms.inference.embedding.limits import (
    VOYAGE_EMBEDDING_DIMENSION,
    VoyageEmbeddingLimits,
)
from kms.inference.embedding.litellm import LiteLLMEmbeddingTransport
from kms.inference.embedding.voyage import voyage_inputs
from kms.inference.tokenization.text import HuggingFaceTextTokenCounter
from kms.media.metadata import ImageMetadataCache


@dataclass(frozen=True, slots=True)
class EmbeddingResources:
    """Embedding implementation, identity, estimator, and hard limits."""

    embedder: Embedder
    model: EmbeddingModel
    estimator: EmbeddingEstimator
    limits: EmbeddingCapacity


@asynccontextmanager
async def embedding_resources(
    settings: Settings,
) -> AsyncIterator[EmbeddingResources]:
    """Select the supported Voyage definition without coupling execution to it."""
    provider_settings = settings.embedding
    if provider_settings.api_key is None:
        raise ValueError('Embedding API key is required')
    counter = HuggingFaceTextTokenCounter(
        provider_settings.tokenizer_path, add_special_tokens=True
    )
    image_metadata = ImageMetadataCache()
    estimator = VoyageEmbeddingEstimator(counter, image_metadata)
    limits = VoyageEmbeddingLimits(image_metadata)
    model = EmbeddingModel(
        model=provider_settings.model,
        dimension=VOYAGE_EMBEDDING_DIMENSION,
    )
    transport = LiteLLMEmbeddingTransport(
        model,
        build_inputs=voyage_inputs,
        request_options={'dimensions': model.dimension, 'truncation': False},
        api_key=provider_settings.api_key.get_secret_value(),
        api_base=provider_settings.api_base,
        timeout_seconds=provider_settings.timeout_seconds,
    )
    embedder = BatchingEmbedder(
        estimator,
        limits,
        transport,
        batch_target_items=provider_settings.batch_target_items,
        batch_target_tokens=provider_settings.batch_target_tokens,
        max_concurrency=provider_settings.max_concurrency,
    )
    yield EmbeddingResources(
        embedder=embedder,
        model=model,
        estimator=estimator,
        limits=limits,
    )
