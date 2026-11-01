"""Hosted and local reranking provider settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RerankerProviderSettings(BaseModel):
    """Provider selection and request budgets for reranking."""

    model_config = ConfigDict(extra='forbid')

    backend: Literal['voyage', 'local'] = 'voyage'
    model: Literal['rerank-3-lite'] = 'rerank-3-lite'
    api_key: SecretStr | None = None
    max_concurrency: int = Field(default=4, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    batch_token_budget: int = Field(default=4096, gt=0)
    tokenizer_path: str | None = None
