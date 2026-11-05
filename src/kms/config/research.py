"""Configuration for awaited teaching-time external research."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from kms.config.inference import VisionInferenceSettings


class TavilySettings(BaseModel):
    """Authentication and request limits for Tavily retrieval."""

    model_config = ConfigDict(extra='forbid')

    api_key: SecretStr | None = None
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = 'basic'
    result_limit: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = Field(default=60.0, gt=0, le=60)


class ResearchSettings(BaseModel):
    """Inference, action, and provider settings for one research call."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    max_tool_calls: int = Field(default=12, ge=0)
    max_concurrency: int = Field(default=4, gt=0)
    gotenberg_url: str = 'http://127.0.0.1:3000'
    download_timeout_seconds: float = Field(default=60.0, gt=0)
    max_download_bytes: int = Field(default=26214400, gt=0)
    max_redirects: int = Field(default=5, ge=0)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
