"""Private state for one acyclic teaching-turn commit workflow."""

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.knowledge.admission import KnowledgeAdmission
from kms2.core.model.teaching.observation import (
    LearnerObservationRecord,
    LearnerObservationWrite,
)
from kms2.core.model.teaching.session import TeachingSessionView
from kms2.core.model.teaching.workflow import (
    TeachingInteractionResult,
    TeachingTurnRequest,
)


class TeachingTurnState(BaseModel):
    """Parent workflow state without transient interaction registries."""

    model_config = ConfigDict(extra='forbid')

    request: TeachingTurnRequest
    learner_history: list[LearnerObservationRecord] = Field(
        default_factory=list
    )
    interaction: TeachingInteractionResult | None = None
    prepared_observations: list[LearnerObservationWrite] = Field(
        default_factory=list
    )
    knowledge_admission: KnowledgeAdmission | None = None
    final_view: TeachingSessionView | None = None
