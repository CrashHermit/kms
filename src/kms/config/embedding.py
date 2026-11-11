"""Voyage multimodal embedding settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class EmbeddingSettings(BaseModel):
    """Provider identity, credentials, and soft batching targets."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['voyage/voyage-multimodal-3.5'] = (
        'voyage/voyage-multimodal-3.5'
    )
    api_key: SecretStr | None = None
    api_base: str | None = None
    max_concurrency: int = Field(default=4, gt=0)
    batch_target_items: int = Field(default=32, gt=0)
    batch_target_tokens: int = Field(default=8192, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    tokenizer_path: str = 'MongoDB/voyage-multimodal-3.5'
