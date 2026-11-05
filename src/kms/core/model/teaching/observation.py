"""Durable, provenance-backed learner observations."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.base import Vertex
from kms.core.model.teaching.positions import Position
from kms.core.model.teaching.question import TeachingQuestionView


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class LearnerObservationKind(StrEnum):
    """Interpretation of one learner's demonstrated understanding."""

    DEMONSTRATED_UNDERSTANDING = 'demonstrated_understanding'
    PARTIAL_UNDERSTANDING = 'partial_understanding'
    GAP = 'gap'
    MISCONCEPTION = 'misconception'


class LearnerObservation(Vertex):
    """Append-only interpretation grounded in learner evidence."""

    kind: LearnerObservationKind
    topic: str
    description: str
    embedding: list[float]
    created_at: datetime


class LearnerObservationDraft(_StrictModel):
    """Model-authored interpretation and its visible learner-turn positions."""

    kind: LearnerObservationKind
    topic: str
    description: str
    evidence_turn_positions: list[Position]


class LearnerObservationWrite(_StrictModel):
    """Prepared observation and resolved learner-turn provenance."""

    observation: LearnerObservation
    evidence_turn_uuids: list[str]


class LearnerObservationEvidence(_StrictModel):
    """Safe learner-turn evidence attached to an observation."""

    turn_uuid: str
    session_uuid: str
    content: str
    created_at: datetime
    question: TeachingQuestionView | None


class LearnerObservationRecord(_StrictModel):
    """Stored learner observation with its directly linked evidence."""

    observation: LearnerObservation
    evidence: list[LearnerObservationEvidence]


class LearnerObservationSearchCandidate(_StrictModel):
    """Owner-scoped cosine candidate before semantic reranking."""

    record: LearnerObservationRecord
    similarity: float


class LearnerObservationSearchHit(LearnerObservationSearchCandidate):
    """Candidate with an actual reranker score and order."""

    relevance_score: float


class LearnerObservationView(_StrictModel):
    """Public observation projection without its embedding."""

    uuid: str
    kind: LearnerObservationKind
    topic: str
    description: str
    created_at: datetime
    evidence: list[LearnerObservationEvidence]


class ExistingLearnerObservationReference(_StrictModel):
    """Reference to a retrieved observation visible to the prediction."""

    kind: Literal['existing']
    position: Position


class NewLearnerObservationReference(_StrictModel):
    """Reference to an observation draft in the current response."""

    kind: Literal['new']
    position: Position


LearnerObservationReference = Annotated[
    ExistingLearnerObservationReference | NewLearnerObservationReference,
    Field(discriminator='kind'),
]


def observation_text(
    kind: LearnerObservationKind,
    topic: str,
    description: str,
) -> str:
    """Build the shared embedding and reranking text for an observation."""
    return f'{kind.value}\n{topic}\n{description}'
