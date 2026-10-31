"""Hosted and local embedding provider settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class EmbeddingProviderSettings(BaseModel):
    """Provider selection and request budgets for embeddings."""

    model_config = ConfigDict(extra='forbid')

    backend: Literal['openrouter', 'local'] = 'openrouter'
    model: Literal['voyageai/voyage-4-large'] = 'voyageai/voyage-4-large'
    api_key: SecretStr | None = None
    max_concurrency: int = Field(default=4, gt=0)
    batch_size: int = Field(default=32, gt=0)
    batch_token_budget: int = Field(default=8192, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    tokenizer_path: str | None = None
