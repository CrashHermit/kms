"""KMS2 configuration with environment-based overrides."""

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRConfig(BaseModel):
    """General OCR materialization settings used by KMS2."""

    model_config = ConfigDict(extra='forbid')

    output_dir: str = 'output'
    render_scale: float = Field(default=1.0, gt=0.0)
    block_crop_scale: float = Field(default=1.0, gt=0.0)


class MistralOCRConfig(BaseModel):
    """Mistral-specific OCR service settings."""

    model_config = ConfigDict(extra='forbid')

    api_key: str = ''
    model: str = 'mistral-ocr-latest'
    url: str = 'https://api.mistral.ai/v1/ocr'


class Settings(BaseSettings):
    """Root KMS2 settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='KMS2_',
        env_nested_delimiter='__',
        extra='forbid',
    )

    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mistral_ocr: MistralOCRConfig = Field(default_factory=MistralOCRConfig)


def load_settings() -> Settings:
    """Builds KMS2 settings from defaults and environment variables."""
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the shared KMS2 settings instance."""
    return load_settings()
