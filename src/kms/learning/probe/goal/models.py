"""Goal clarification inference contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.learning.session.models import SessionMessage


class ClarificationQuestion(BaseModel):
    """A question needed to clarify the learner's goal."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['clarify'] = 'clarify'
    question: str


class ClarifiedGoal(BaseModel):
    """A goal ready for decomposition into search queries."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['ready'] = 'ready'
    goal: str


GoalDecision = Annotated[
    ClarificationQuestion | ClarifiedGoal, Field(discriminator='kind')
]


class GoalClarificationInput(BaseModel):
    """Ordered exchange used to interpret the goal."""

    model_config = ConfigDict(extra='forbid')

    messages: list[SessionMessage]
