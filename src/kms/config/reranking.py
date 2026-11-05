"""IONOS multimodal reranking settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RerankingSettings(BaseModel):
    """Provider identity, credentials, and request budgets."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['Qwen/Qwen3-VL-Reranker-8B'] = 'Qwen/Qwen3-VL-Reranker-8B'
    api_key: SecretStr | None = None
    max_concurrency: int = Field(default=4, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    batch_token_budget: int = Field(default=32768, gt=0)
    tokenizer_path: str = 'Qwen/Qwen3-VL-Reranker-8B'
