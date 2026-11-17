"""Limits for retrieving knowledge candidates and connected context."""

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSearchSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    candidate_limit: int = Field(default=64, gt=0)
    result_limit: int = Field(default=32, gt=0)
    argument_limit: int = Field(default=64, gt=0)
