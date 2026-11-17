"""Root environment-aware KMS2 configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kms.config.embedding import EmbeddingSettings
from kms.config.llm import LLMSettings
from kms.config.runtime import LocalModelRuntimeSettings
from kms.config.services import DatabaseSettings, OCRSettings, TrainingSettings
from kms.knowledge.correctness.settings import CorrectnessSettings
from kms.knowledge.extraction.settings import UnitExtractionSettings
from kms.knowledge.identity.settings import IdentitySettings
from kms.knowledge.reasoning.settings import ReasoningSettings
from kms.knowledge.retrieval.settings import KnowledgeSearchSettings
from kms.knowledge.sufficiency.settings import SufficiencySettings
from kms.learning.memory.settings import MemorySettings
from kms.learning.probe.goal.settings import GoalClarificationSettings
from kms.learning.probe.targets.settings import TargetDecompositionSettings
from kms.learning.session.planning.settings import LearningPlanningSettings
from kms.learning.session.settings import LearningSessionSettings
from kms.learning.task.generation.settings import TaskGenerationSettings
from kms.learning.task.grading.settings import TaskGradingSettings
from kms.learning.teaching.generation.settings import TeachingGenerationSettings
from kms.retrieval.selection.settings import SearchSelectionSettings
from kms.source.capture.settings import SourceCaptureSettings
from kms.source.discovery.settings import SourceDiscoverySettings
from kms.source.processing.settings import SourceProcessingSettings
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
    target_decomposition: TargetDecompositionSettings = Field(
        default_factory=TargetDecompositionSettings
    )
    learning_session: LearningSessionSettings = Field(
        default_factory=LearningSessionSettings
    )
    task_generation: TaskGenerationSettings = Field(
        default_factory=TaskGenerationSettings
    )
    task_grading: TaskGradingSettings = Field(
        default_factory=TaskGradingSettings
    )
    teaching_generation: TeachingGenerationSettings = Field(
        default_factory=TeachingGenerationSettings
    )
    learning_planning: LearningPlanningSettings = Field(
        default_factory=LearningPlanningSettings
    )
    memory: MemorySettings = Field(default_factory=MemorySettings)
    knowledge_extraction: UnitExtractionSettings = Field(
        default_factory=UnitExtractionSettings
    )
    knowledge_search: KnowledgeSearchSettings = Field(
        default_factory=KnowledgeSearchSettings
    )
    knowledge_reasoning: ReasoningSettings = Field(
        default_factory=ReasoningSettings
    )
    knowledge_identity: IdentitySettings = Field(
        default_factory=IdentitySettings
    )
    knowledge_sufficiency: SufficiencySettings = Field(
        default_factory=SufficiencySettings
    )
    knowledge_correctness: CorrectnessSettings = Field(
        default_factory=CorrectnessSettings
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
