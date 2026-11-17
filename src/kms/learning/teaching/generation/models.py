"""Requests, drafts, and execution state for grounded teaching."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from kms.knowledge.models import KnowledgeContext
from kms.learning.materials import MaterialDraft
from kms.learning.references import GroundingReferences, LearningTarget
from kms.learning.session.models import SessionContext
from kms.learning.task.models import Task
from kms.source.retrieval.models import SourceUnitWindow


class TeachingGenerationRequest(BaseModel):
    """Owned context for teaching one supplied learning objective."""

    model_config = ConfigDict(extra='forbid')

    session: SessionContext
    understanding: str
    objective: str
    target: LearningTarget
    knowledge: KnowledgeContext
    evidence: list[SourceUnitWindow] = Field(default_factory=list)
    assessment_uuids: list[str] = Field(default_factory=list)
    solution_task: Task | None = None


class TeachingDraft(BaseModel):
    """Generated native material and its grounding references."""

    model_config = ConfigDict(extra='forbid')

    material: MaterialDraft
    grounding: GroundingReferences


class TeachingState(TypedDict):
    """Owner-scoped state for generating and persisting teaching."""

    user_uuid: str
    session_uuid: str
    request: TeachingGenerationRequest
