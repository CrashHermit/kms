"""Teaching-session aggregates and interaction commit contracts."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from kms2.core.model.base import Vertex
from kms2.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
)
from kms2.core.model.teaching.probe import ProbeAssessment
from kms2.core.model.teaching.question import (
    TeachingQuestion,
    TeachingQuestionView,
)
from kms2.core.model.teaching.turn import TeachingTurn


class TeachingStatus(StrEnum):
    """Lifecycle state that determines the next session operation."""

    AWAITING_TEACHER = 'awaiting_teacher'
    AWAITING_LEARNER = 'awaiting_learner'
    ENDED = 'ended'


class TeachingPhase(StrEnum):
    """Current pedagogical phase selected by the teaching workflow."""

    PROBING = 'probing'
    PLANNING = 'planning'
    TEACHING = 'teaching'


class TeachingSession(Vertex):
    """Goal-driven teaching session lifecycle state."""

    goal: str
    status: TeachingStatus
    phase: TeachingPhase
    revision: int
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None


class TeachingTurnRecord(BaseModel):
    """One turn plus its relationship-derived teaching context."""

    model_config = ConfigDict(extra='forbid')

    turn: TeachingTurn
    question: TeachingQuestion | None
    question_step_uuid: str | None
    responds_to_question_uuid: str | None
    presented_plan_uuid: str | None
    accepted_plan_uuid: str | None


class TeachingSessionSnapshot(BaseModel):
    """Private session aggregate used by teaching orchestration."""

    model_config = ConfigDict(extra='forbid')

    session: TeachingSession
    turns: list[TeachingTurnRecord]
    current_assessment: ProbeAssessment | None
    current_plan: TeachingPlan | None
    current_plan_acceptance: TeachingPlanAcceptance | None
    current_step_uuid: str | None
    pending_question_uuid: str | None


class TeachingTurnView(BaseModel):
    """Learner-safe turn projection with relationship context."""

    model_config = ConfigDict(extra='forbid')

    turn: TeachingTurn
    question: TeachingQuestionView | None
    question_step_uuid: str | None
    responds_to_question_uuid: str | None
    presented_plan_uuid: str | None
    accepted_plan_uuid: str | None


class TeachingSessionView(BaseModel):
    """Learner-safe teaching session projection."""

    model_config = ConfigDict(extra='forbid')

    session: TeachingSession
    turns: list[TeachingTurnView]
    current_assessment: ProbeAssessment | None
    current_plan: TeachingPlan | None
    current_plan_acceptance: TeachingPlanAcceptance | None
    current_step_uuid: str | None
    pending_question_uuid: str | None


class TeachingTurnCommit(BaseModel):
    """Complete intended state for one persisted teacher turn."""

    model_config = ConfigDict(extra='forbid')

    phase: TeachingPhase
    current_assessment_uuid: str | None
    current_plan_uuid: str | None
    current_step_uuid: str | None
    pending_question_uuid: str | None
    new_assessment: ProbeAssessment | None = None
    new_plan: TeachingPlan | None = None
    new_question: TeachingQuestion | None = None
    new_question_step_uuid: str | None = None
    plan_acceptance: TeachingPlanAcceptance | None = None
