"""State of an interactive goal clarification."""

from pydantic import BaseModel, Field

from kms.learning.probe.goal.models import ClarificationQuestion, GoalDecision
from kms.learning.session.models import SessionMessage


class GoalClarificationState(BaseModel):
    """Session identity, local exchange, and current goal decision."""

    session_uuid: str
    user_uuid: str
    messages: list[SessionMessage] = Field(default_factory=list)
    decision: GoalDecision = Field(
        default_factory=lambda: ClarificationQuestion(
            question='What would you like to learn?'
        )
    )
