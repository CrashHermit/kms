"""FSRS policy settings for learner memory."""

from pydantic import BaseModel, ConfigDict, Field


class MemorySettings(BaseModel):
    """Configure the desired FSRS retention rate."""

    model_config = ConfigDict(extra='forbid')

    desired_retention: float = Field(default=0.9, gt=0, lt=1)
