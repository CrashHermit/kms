"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms.config.embedding import EmbeddingSettings
from kms.config.llm import LLMSettings
from kms.config.runtime import LocalModelRuntimeSettings
from kms.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms.learning.probe.goal.settings import GoalClarificationSettings
from kms.source.capture.settings import SourceCaptureSettings
from kms.source.discovery.settings import SourceDiscoverySettings
from kms.source.processing.settings import SourceProcessingSettings
from kms.source.retrieval.selection.settings import SearchSelectionSettings
from kms.source.retrieval.settings import SourceSearchSettings

_DEFAULT_ENV = Path(__file__).resolve().parents[3] / '.env'


class Settings(BaseSettings):
    """KMS2 runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix='KMS_',
        env_nested_delimiter='__',
        env_file=_DEFAULT_ENV,
        nested_model_default_partial_update=True,
    )

    source_processing: SourceProcessingSettings = Field(
        default_factory=SourceProcessingSettings
    )

    source_search: SourceSearchSettings = Field(
        default_factory=SourceSearchSettings
    )
    search_selection: SearchSelectionSettings = Field(
        default_factory=SearchSelectionSettings
    )
    goal_clarification: GoalClarificationSettings = Field(
        default_factory=GoalClarificationSettings
    )
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    source_discovery: SourceDiscoverySettings = Field(
        default_factory=SourceDiscoverySettings
    )
    source_capture: SourceCaptureSettings = Field(
        default_factory=SourceCaptureSettings
    )
