"""Inputs and state for learning-session planning."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from kms.learning.references import LearningTarget
from kms.learning.session.models import SessionContext


class LearningPlanningRequest(BaseModel):
    """The selected session context and optional target anchor."""

    model_config = ConfigDict(extra='forbid')

    session: SessionContext
    target: LearningTarget | None = None


class PlanningState(TypedDict):
    """Identity and selected context supplied to a planning node."""

    user_uuid: str
    session_uuid: str
    request: LearningPlanningRequest
