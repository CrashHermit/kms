"""Private state for one complete teaching workflow."""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.knowledge.admission import KnowledgeAdmission
from kms.core.model.research.inference import (
    ResearchAction,
    ResearchFailure,
    ResearchResult,
    ResearchSearchResult,
    ResearchTask,
    ResearchToolObservation,
)
from kms.core.model.source import SourceRevisionReference
from kms.core.model.teaching.inference import TeachingAction, TeachingContext
from kms.core.model.teaching.observation import (
    LearnerObservationRecord,
    LearnerObservationWrite,
)
from kms.core.model.teaching.session import TeachingSessionView
from kms.core.model.teaching.workflow import (
    TeachingInteractionResult,
    TeachingTurnRequest,
)
from kms.core.model.teaching.workspace import TeachingEvidenceWorkspace


class TeachingTurnState(BaseModel):
    """Carry durable turn data and transient evidence through one flat graph."""

    model_config = ConfigDict(extra='forbid')

    request: TeachingTurnRequest
    learner_history: list[LearnerObservationRecord] = Field(
        default_factory=list
    )
    workspace: TeachingEvidenceWorkspace = Field(
        default_factory=TeachingEvidenceWorkspace
    )
    tool_calls: int = 0
    latest_action: TeachingAction | ResearchAction | None = None
    latest_context: TeachingContext | None = None
    visible_turn_indexes: list[int] = Field(default_factory=list)
    visible_observation_indexes: list[int] = Field(default_factory=list)
    visible_citation_receipt_indexes: list[int] = Field(default_factory=list)
    interaction: TeachingInteractionResult | None = None
    prepared_observations: list[LearnerObservationWrite] = Field(
        default_factory=list
    )
    knowledge_admission: KnowledgeAdmission | None = None
    final_view: TeachingSessionView | None = None
    research_task: ResearchTask | None = None
    research_max_tool_calls: int = 0
    search_results: list[ResearchSearchResult] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)
    submitted_urls: set[str] = Field(default_factory=set)
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    persisted_source_revisions: list[SourceRevisionReference] = Field(
        default_factory=list
    )
    research_observations: list[ResearchToolObservation] = Field(
        default_factory=list
    )
    final_research_result: ResearchResult | None = None
    research_tool_calls: int = 0
    research_outcome_position: int | None = None
