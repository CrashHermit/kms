"""Teaching-probe assessment graph models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.base import Vertex


class ProbeStrand(BaseModel):
    """One topic-level understanding and limitation observed during probing."""

    model_config = ConfigDict(extra='forbid')

    topic: str
    demonstrated_understanding: str
    observed_limit: str
    gaps: list[str]
    misconceptions: list[str]


class ProbeAssessment(Vertex):
    """One immutable assessment revision for a teaching session."""

    summary: str
    strands: list[ProbeStrand]
    observation_uuids: list[str] = Field(default_factory=list)
    created_at: datetime
