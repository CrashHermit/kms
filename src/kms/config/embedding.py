"""Voyage multimodal embedding settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class EmbeddingSettings(BaseModel):
    """Provider identity, credentials, and request budgets."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['voyage/voyage-multimodal-3.5'] = (
        'voyage/voyage-multimodal-3.5'
    )
    api_key: SecretStr | None = None
    max_concurrency: int = Field(default=4, gt=0)
    batch_size: int = Field(default=32, gt=0, le=1000)
    batch_token_budget: int = Field(default=8192, gt=0, le=320000)
    timeout_seconds: float = Field(default=120.0, gt=0)
    tokenizer_path: str = 'MongoDB/voyage-multimodal-3.5'
