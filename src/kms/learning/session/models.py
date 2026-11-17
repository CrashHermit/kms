"""Typed learning-session timelines and selected inference context."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.models import Vertex
from kms.learning.references import LearningTarget
from kms.learning.task.models import (
    Task,
    TaskAssessment,
    TaskResponse,
)
from kms.learning.teaching.models import Teaching


class SessionMessage(BaseModel):
    """One message in the learner's session conversation."""

    model_config = ConfigDict(extra='forbid')

    role: Literal['assistant', 'user']
    content: str


class AssessmentTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['assess'] = 'assess'
    understanding: str
    explanation: str
    assessment_uuids: list[str]
    target: LearningTarget | None = None


class TeachingTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['teach'] = 'teach'
    understanding: str
    objective: str
    explanation: str
    assessment_uuids: list[str]
    target: LearningTarget


class LearningFinished(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['finish'] = 'finish'
    summary: str
    assessment_uuids: list[str]


LearningDecision = Annotated[
    AssessmentTarget | TeachingTarget | LearningFinished,
    Field(discriminator='kind'),
]


class LearningDecisionRecord(Vertex):
    kind: Literal['decision'] = 'decision'
    decision: LearningDecision


SessionStep = Annotated[
    Task | Teaching | LearningDecisionRecord, Field(discriminator='kind')
]


class LearningSession(Vertex):
    goal: str
    targets: list[str]
    status: Literal['active', 'completed'] = 'active'


class SessionHistory(BaseModel):
    model_config = ConfigDict(extra='forbid')

    session: LearningSession
    steps: list[SessionStep]
    responses: list[TaskResponse]
    assessments: list[TaskAssessment]


class SessionContext(BaseModel):
    model_config = ConfigDict(extra='forbid')

    goal: str
    targets: list[str]
    steps: list[SessionStep]
    responses: list[TaskResponse]
    assessments: list[TaskAssessment]
