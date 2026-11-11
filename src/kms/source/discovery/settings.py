"""Configuration for source discovery."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from kms.config.inference import TextInferenceSettings


class TavilySettings(BaseModel):
    """Authentication and request limits for Tavily retrieval."""

    model_config = ConfigDict(extra='forbid')

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(''))
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = 'basic'
    result_limit: int = Field(default=3, ge=1, le=20)
    timeout_seconds: float = Field(default=60.0, gt=0, le=60)


class SourceDiscoverySettings(BaseModel):
    """Inference and provider settings for one discovery call."""

    model_config = ConfigDict(extra='forbid')

    inference: TextInferenceSettings = Field(
        default_factory=lambda: TextInferenceSettings(num_retries=0)
    )
    tavily: TavilySettings = Field(default_factory=TavilySettings)
