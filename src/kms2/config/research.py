"""Configuration for awaited teaching-time external research."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from kms2.config.inference import TextInferenceSettings, VisionInferenceSettings
from kms2.config.source_processing import FormattingSettings


class TavilySettings(BaseModel):
    """Authentication and request limits for Tavily retrieval."""

    model_config = ConfigDict(extra='forbid')

    api_key: SecretStr | None = None
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = 'basic'
    result_limit: int = Field(default=5, ge=1, le=20)
    extract_depth: Literal['basic', 'advanced'] = 'advanced'
    timeout_seconds: float = Field(default=60.0, gt=0, le=60)


def _formatting_settings() -> FormattingSettings:
    """Build representation formatting without request retries."""
    return FormattingSettings(inference=TextInferenceSettings(num_retries=0))


class ResearchSettings(BaseModel):
    """Budgets, media limits, and provider settings for one research call."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    max_tool_calls: int = Field(default=12, ge=0)
    max_context_images: int = Field(default=4, ge=0)
    max_context_image_bytes: int = Field(default=26214400, gt=0)
    max_context_image_pixels: int = Field(default=40000000, gt=0)
    formatting: FormattingSettings = Field(default_factory=_formatting_settings)
    passage_token_budget: int = Field(default=1024, gt=0)
    catalog_page_size: int = Field(default=12, gt=0)
    pdf_search_result_limit: int = Field(default=12, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    download_timeout_seconds: float = Field(default=60.0, gt=0)
    max_download_bytes: int = Field(default=26214400, gt=0)
    max_image_pixels: int = Field(default=40000000, gt=0)
    max_redirects: int = Field(default=5, ge=0)
    source_directory: Path = Path('output/research/sources')
    tavily: TavilySettings = Field(default_factory=TavilySettings)
