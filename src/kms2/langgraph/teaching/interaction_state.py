"""Private state for one bounded instructional decision loop."""

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.teaching.inference import TeachingAction
from kms2.core.model.teaching.workflow import (
    TeachingInteractionRequest,
    TeachingInteractionResult,
)
from kms2.core.model.teaching.workspace import TeachingEvidenceWorkspace


class TeachingInteractionState(BaseModel):
    """Transient action state that is discarded after one teacher response."""

    model_config = ConfigDict(extra='forbid')

    request: TeachingInteractionRequest
    workspace: TeachingEvidenceWorkspace = Field(
        default_factory=TeachingEvidenceWorkspace
    )
    tool_calls: int = 0
    latest_action: TeachingAction | None = None
    visible_turn_indexes: list[int] = Field(default_factory=list)
    visible_observation_indexes: list[int] = Field(default_factory=list)
    visible_citation_receipt_indexes: list[int] = Field(default_factory=list)
    result: TeachingInteractionResult | None = None
