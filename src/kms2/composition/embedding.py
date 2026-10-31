"""Composition for the selected embedding backend."""

from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.core.windowing import TextTokenCounter
from kms2.inference.embedding import OpenRouterEmbeddingClient
from kms2.local_models.runtime import LocalModelRuntime

VOYAGE_EMBEDDING_DIMENSION = 1024


def embedding_dimension(settings: Settings) -> int:
    """Return the vector dimension required by the selected backend."""
    if settings.embedding.backend == 'openrouter':
        return VOYAGE_EMBEDDING_DIMENSION
    if settings.embedding.backend == 'local':
        return settings.local_models.embedding.model.dimension
    raise ValueError(
        f'Unsupported embedding backend: {settings.embedding.backend}'
    )


def build_embedding_client(
    settings: Settings,
    local_models: LocalModelRuntime,
    token_counter: TextTokenCounter,
) -> EmbeddingClient:
    """Build the chosen client without starting local model processes."""
    if settings.embedding.backend == 'local':
        return local_models
    if settings.embedding.backend == 'openrouter':
        return OpenRouterEmbeddingClient(settings.embedding, token_counter)
    raise ValueError(
        f'Unsupported embedding backend: {settings.embedding.backend}'
    )
