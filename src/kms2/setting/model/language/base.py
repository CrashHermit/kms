"""Shared configuration for local and remote language models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LanguageModelCapability(StrEnum):
    """Input capabilities provided by a language model."""

    TEXT = 'text'
    VISION = 'vision'


class LanguageModelBase(BaseModel):
    """Shared settings for local and remote language models."""

    model_config = ConfigDict(extra='forbid')

    model: str = Field(min_length=1)
    capabilities: set[LanguageModelCapability] = {LanguageModelCapability.TEXT}
    api_base: str | None = None
    api_key: str | None = None
    provider_options: dict[str, object] = Field(default_factory=dict)

    temperature: float = 0.0
    max_tokens: int = Field(default=8192, gt=0)
    num_retries: int = Field(default=3, ge=0)
    cache: bool = True
