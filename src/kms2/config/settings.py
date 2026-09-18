"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.config.semantic import SemanticSettings
from kms2.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms2.config.source import SourceSettings

_DEFAULT_ENV = Path(__file__).resolve().parents[3] / '.env'


class Settings(BaseSettings):
    """KMS2 runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix='KMS2_',
        env_nested_delimiter='__',
        env_file=_DEFAULT_ENV,
    )

    source: SourceSettings = Field(default_factory=SourceSettings)
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
