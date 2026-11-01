"""Private state for one isolated evidence action."""

from pydantic import BaseModel, ConfigDict

from kms2.core.model.teaching.workflow import (
    EvidenceExecutionRequest,
    EvidenceExecutionResult,
)


class EvidenceAcquisitionState(BaseModel):
    model_config = ConfigDict(extra='forbid')

    request: EvidenceExecutionRequest
    result: EvidenceExecutionResult | None = None
