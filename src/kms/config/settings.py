"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms.config.embedding import EmbeddingSettings
from kms.config.knowledge import KnowledgeSettings
from kms.config.llm import LLMSettings
from kms.config.reranking import RerankingSettings
from kms.config.research import ResearchSettings
from kms.config.runtime import LocalModelRuntimeSettings
from kms.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms.config.source_learning import SourceLearningSettings
from kms.config.source_pedagogical import SourcePedagogicalSettings
from kms.config.source_processing import SourceProcessingSettings
from kms.config.source_search import (
    SearchSelectionSettings,
    SourceSearchSettings,
)
from kms.config.source_semantic import SourceSemanticSettings
from kms.config.teaching import TeachingSettings

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
    source_pedagogical: SourcePedagogicalSettings = Field(
        default_factory=SourcePedagogicalSettings
    )
    source_semantic: SourceSemanticSettings = Field(
        default_factory=SourceSemanticSettings
    )
    source_learning: SourceLearningSettings = Field(
        default_factory=SourceLearningSettings
    )

    source_search: SourceSearchSettings = Field(
        default_factory=SourceSearchSettings
    )
    search_selection: SearchSelectionSettings = Field(
        default_factory=SearchSelectionSettings
    )
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankingSettings = Field(default_factory=RerankingSettings)
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    research: ResearchSettings = Field(default_factory=ResearchSettings)

    teaching: TeachingSettings = Field(default_factory=TeachingSettings)

    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
