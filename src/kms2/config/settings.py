"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms2.config.global_semantic import GlobalSemanticSettings
from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms2.config.source_learning import SourceLearningSettings
from kms2.config.source_processing import SourceProcessingSettings
from kms2.config.source_semantic import SourceSemanticSettings

_DEFAULT_ENV = Path(__file__).resolve().parents[3] / '.env'


class Settings(BaseSettings):
    """KMS2 runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix='KMS2_',
        env_nested_delimiter='__',
        env_file=_DEFAULT_ENV,
    )

    source_processing: SourceProcessingSettings = Field(
        default_factory=SourceProcessingSettings
    )
    source_semantic: SourceSemanticSettings = Field(
        default_factory=SourceSemanticSettings
    )
    source_learning: SourceLearningSettings = Field(
        default_factory=SourceLearningSettings
    )
    global_semantic: GlobalSemanticSettings = Field(
        default_factory=GlobalSemanticSettings
    )
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
