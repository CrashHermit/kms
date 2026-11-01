"""Explicit instructional operations and their model-facing drafts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.teaching.observation import (
    LearnerObservationDraft,
    LearnerObservationReference,
)
from kms2.core.model.teaching.positions import Position
from kms2.core.model.teaching.probe import ProbeStrand
from kms2.core.model.teaching.question import (
    TeachingQuestionKind,
    TeachingResponseSpecification,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TeachingAssessmentDraft(_StrictModel):
    summary: str
    strands: list[ProbeStrand]
    observation_references: list[LearnerObservationReference] = Field(
        default_factory=list
    )


class TeachingPlanStepDraft(_StrictModel):
    title: str
    objective: str
    prerequisite_positions: list[Position]


class TeachingPlanDraft(_StrictModel):
    summary: str
    steps: list[TeachingPlanStepDraft]


class TeachingQuestionDraft(_StrictModel):
    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseSpecification


class ContinueProbing(_StrictModel):
    kind: Literal['continue_probing']
    question: TeachingQuestionDraft
    assessment: TeachingAssessmentDraft | None = None


class CompleteProbing(_StrictModel):
    kind: Literal['complete_probing']
    assessment: TeachingAssessmentDraft


class ProposePlan(_StrictModel):
    kind: Literal['propose_plan']
    plan: TeachingPlanDraft
    question: TeachingQuestionDraft


class AcceptPlan(_StrictModel):
    kind: Literal['accept_plan']
    learner_turn_position: Position
    step_position: Position


class TeachStep(_StrictModel):
    kind: Literal['teach_step']
    step_position: Position
    question: TeachingQuestionDraft
    assessment: TeachingAssessmentDraft | None = None


class RepairCheck(_StrictModel):
    kind: Literal['repair_check']
    question: TeachingQuestionDraft
    assessment: TeachingAssessmentDraft | None = None


class Hint(_StrictModel):
    kind: Literal['hint']


class Consolidate(_StrictModel):
    kind: Literal['consolidate']
    assessment: TeachingAssessmentDraft | None = None


class ReopenProbing(_StrictModel):
    kind: Literal['reopen_probing']
    question: TeachingQuestionDraft
    assessment: TeachingAssessmentDraft | None = None


class ReturnToPlanning(_StrictModel):
    kind: Literal['return_to_planning']
    question: TeachingQuestionDraft | None = None


class Clarify(_StrictModel):
    kind: Literal['clarify']
    question: TeachingQuestionDraft | None = None


TeachingOperation = Annotated[
    ContinueProbing
    | CompleteProbing
    | ProposePlan
    | AcceptPlan
    | TeachStep
    | RepairCheck
    | Hint
    | Consolidate
    | ReopenProbing
    | ReturnToPlanning
    | Clarify,
    Field(discriminator='kind'),
]


class TeachingAssessmentContext(_StrictModel):
    """Learner-safe assessment projection without observation identities."""

    summary: str
    strands: list[ProbeStrand]


class RespondAction(_StrictModel):
    """Publish one learner-facing response and its explicit operation."""

    kind: Literal['respond'] = 'respond'
    content: str
    content_kind: Literal['instruction', 'coordination']
    operation: TeachingOperation
    learner_observations: list[LearnerObservationDraft] = Field(
        default_factory=list
    )
    cited_evidence_positions: list[Position] = Field(default_factory=list)
