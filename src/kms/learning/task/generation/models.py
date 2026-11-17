"""Grounded free-response task requests and output drafts."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from kms.knowledge.models import KnowledgeContext
from kms.learning.materials import MaterialDraft
from kms.learning.references import (
    ArgumentTarget,
    GroundingReferences,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.models import SessionContext
from kms.learning.task.models import Task
from kms.source.retrieval.models import SourceUnitWindow


class NewSolutionTarget(BaseModel):
    """Ask the generated task's newly committed solution."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['new_solution'] = 'new_solution'


TaskTargetRequest = Annotated[
    PropositionTarget | ArgumentTarget | SolutionTarget | NewSolutionTarget,
    Field(discriminator='kind'),
]


class TaskGenerationRequest(BaseModel):
    """The owned context and one caller-selected target for task generation."""

    model_config = ConfigDict(extra='forbid')

    session: SessionContext
    understanding: str
    knowledge: KnowledgeContext
    evidence: list[SourceUnitWindow] = Field(default_factory=list)
    target: TaskTargetRequest = Field(default_factory=NewSolutionTarget)
    solution_task: Task | None = None


class SolutionDraft(BaseModel):
    """Hidden reference solution and assessment criteria."""

    model_config = ConfigDict(extra='forbid')

    material: MaterialDraft
    assessment_criteria: list[str]
    grounding: GroundingReferences


class TaskDraft(BaseModel):
    """Complete open-response task and its hidden solution."""

    model_config = ConfigDict(extra='forbid')

    prompt: MaterialDraft
    grounding: GroundingReferences
    solution: SolutionDraft


class TaskState(TypedDict):
    """Owner-scoped state for generating and persisting one task."""

    user_uuid: str
    session_uuid: str
    request: TaskGenerationRequest
