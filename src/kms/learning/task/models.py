"""Durable tasks, reference solutions, learner attempts, and assessments."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.models import Vertex
from kms.learning.references import GroundingReferences, LearningTarget
from kms.media.models import Material


class Confidence(StrEnum):
    """Learner-reported confidence, not a correctness or scheduling grade."""

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class ReviewRating(StrEnum):
    """Review outcome available to a future memory scheduler."""

    AGAIN = 'again'
    HARD = 'hard'
    GOOD = 'good'
    EASY = 'easy'


class Solution(Vertex):
    """A reference solution and criteria, not the only acceptable method."""

    material: Material
    assessment_criteria: list[str]
    grounding: GroundingReferences


class Task(Vertex):
    """An immutable assessment task committed with its hidden solution."""

    kind: Literal['task'] = 'task'
    understanding: str
    prompt: Material
    target: LearningTarget
    grounding: GroundingReferences
    solution: Solution


class PresentedTask(BaseModel):
    """Learner-facing content without the reference solution or criteria."""

    model_config = ConfigDict(extra='forbid')

    task_uuid: str
    prompt: Material


class FreeAnswer(BaseModel):
    """One multimodal field containing the learner's answer and any work."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['answer'] = 'answer'
    material: Material
    confidence: Confidence


class UnknownTaskAnswer(BaseModel):
    """An explicit unknown answer that may include partial work."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['unknown'] = 'unknown'
    material: Material = Field(default_factory=Material)
    confidence: None = None


TaskAnswer = Annotated[
    FreeAnswer | UnknownTaskAnswer, Field(discriminator='kind')
]


class TaskResponse(Vertex):
    """A learner attempt whose task association is stored as a relationship."""

    task_uuid: str
    answer: TaskAnswer


class LearningObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: str
    evidence: str
    uncertainty: str
    follow_up: str


class ResponseInterpretation(BaseModel):
    """Evidence about reasoning, distinct from final-answer correctness."""

    model_config = ConfigDict(extra='forbid')

    reasoning_verdict: Literal[
        'sound', 'mixed', 'unsound', 'insufficient_evidence'
    ]
    explanation: str
    demonstrated_understandings: list[str]
    gaps: list[LearningObservation]
    possible_misconceptions: list[LearningObservation]
    unresolved_questions: list[str]


class TaskGrade(BaseModel):
    """Model assessment of an attempt, not an update to memory state."""

    model_config = ConfigDict(extra='forbid')

    correctness: Literal['correct', 'incorrect', 'unknown']
    interpretation: ResponseInterpretation
    review_rating: ReviewRating


class TaskAssessment(Vertex):
    """A persisted grade associated with the actual recorded response."""

    response_uuid: str
    grade: TaskGrade
