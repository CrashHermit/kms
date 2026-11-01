"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms2.config.embedding import EmbeddingProviderSettings
from kms2.config.knowledge import KnowledgeSettings
from kms2.config.llm import LLMSettings
from kms2.config.reranking import RerankerProviderSettings
from kms2.config.research import ResearchSettings
from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms2.config.source_learning import SourceLearningSettings
from kms2.config.source_pedagogical import SourcePedagogicalSettings
from kms2.config.source_processing import SourceProcessingSettings
from kms2.config.source_search import SourceSearchSettings
from kms2.config.source_semantic import SourceSemanticSettings
from kms2.config.teaching import TeachingSettings

_DEFAULT_ENV = Path(__file__).resolve().parents[3] / '.env'


class Settings(BaseSettings):
    """KMS2 runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix='KMS2_',
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
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingProviderSettings = Field(
        default_factory=EmbeddingProviderSettings
    )
    reranker: RerankerProviderSettings = Field(
        default_factory=RerankerProviderSettings
    )
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    research: ResearchSettings = Field(default_factory=ResearchSettings)

    teaching: TeachingSettings = Field(default_factory=TeachingSettings)

    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
