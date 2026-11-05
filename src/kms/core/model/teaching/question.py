"""Teaching-question models and learner-safe question projections."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.base import Vertex
from kms.core.model.teaching.positions import Position


class TeachingQuestionKind(StrEnum):
    """Interaction role of a teacher question."""

    PROBE = 'probe'
    PLAN_APPROVAL = 'plan_approval'
    PREDICT = 'predict'
    CHECK = 'check'
    CLARIFICATION = 'clarification'


class TeachingOption(BaseModel):
    """One learner-visible multiple-choice option."""

    model_config = ConfigDict(extra='forbid')

    key: str
    content: str


class FreeResponseSpecification(BaseModel):
    """Private specification for an open-response teaching question."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['free_response'] = 'free_response'
    reference_answer: str | None = None


class MultipleChoiceSpecification(BaseModel):
    """Private specification for a multiple-choice teaching question."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['multiple_choice'] = 'multiple_choice'
    options: list[TeachingOption]
    correct_option_key: str


TeachingResponseSpecification = Annotated[
    FreeResponseSpecification | MultipleChoiceSpecification,
    Field(discriminator='kind'),
]


class TeachingQuestion(Vertex):
    """One immutable teacher question with private grading metadata."""

    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseSpecification
    created_at: datetime


class FreeResponseView(BaseModel):
    """Learner-safe projection of an open-response specification."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['free_response'] = 'free_response'


class MultipleChoiceView(BaseModel):
    """Learner-safe projection of a multiple-choice specification."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['multiple_choice'] = 'multiple_choice'
    options: list[TeachingOption]


TeachingResponseView = Annotated[
    FreeResponseView | MultipleChoiceView,
    Field(discriminator='kind'),
]


class TeachingQuestionView(BaseModel):
    """Learner-safe question projection without answer metadata."""

    model_config = ConfigDict(extra='forbid')

    uuid: str
    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseView
    created_at: datetime


class TeachingQuestionContext(BaseModel):
    """Current-session question projection with backend-derived step linkage."""

    model_config = ConfigDict(extra='forbid')

    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseSpecification
    step_position: Position | None


def question_view(question: TeachingQuestion) -> TeachingQuestionView:
    """Project a typed question response without its private answer."""
    response = question.response
    if isinstance(response, FreeResponseSpecification):
        response_view = FreeResponseView()
    else:
        response_view = MultipleChoiceView(options=response.options)
    return TeachingQuestionView(
        uuid=question.uuid,
        kind=question.kind,
        prompt=question.prompt,
        response=response_view,
        created_at=question.created_at,
    )
