"""Strict boundary contracts for isolated evidence actions."""

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.teaching.inference import (
    EvidenceToolAction,
    TeachingToolObservation,
)
from kms2.core.model.teaching.observation import (
    LearnerObservationRecord,
)
from kms2.core.model.teaching.operation import RespondAction
from kms2.core.model.teaching.session import (
    TeachingSessionSnapshot,
)
from kms2.core.model.teaching.workspace import TeachingEvidenceWorkspace


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TeachingTurnRequest(_StrictModel):
    """One expected durable session revision and caller-selected source scope."""

    user_uuid: str
    session_uuid: str
    expected_revision: int
    snapshot: TeachingSessionSnapshot
    source_uuids: list[str] | None


class TeachingInteractionRequest(_StrictModel):
    """One bounded decision workflow and its retrieved learner history."""

    turn: TeachingTurnRequest
    learner_history: list[LearnerObservationRecord] = Field(
        default_factory=list
    )


class TeachingInteractionResult(_StrictModel):
    """Typed response, private evidence workspace, and visible-position maps."""

    response: RespondAction
    workspace: TeachingEvidenceWorkspace
    visible_turn_indexes: list[int]
    visible_observation_indexes: list[int]
    visible_citation_receipt_indexes: list[int]


class EvidenceExecutionRequest(_StrictModel):
    """One authorized tool action and its session-private workspace."""

    user_uuid: str
    source_uuids: list[str] | None
    action: EvidenceToolAction
    workspace: TeachingEvidenceWorkspace


class EvidenceExecutionResult(_StrictModel):
    """The mutated workspace and single observation from one completed action."""

    workspace: TeachingEvidenceWorkspace
    observation: TeachingToolObservation
