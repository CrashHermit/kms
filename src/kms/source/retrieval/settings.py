"""Configuration for source retrieval."""

from pydantic import BaseModel, Field


class SourceSearchSettings(BaseModel):
    """Candidate and final-result limits for source search."""

    candidate_limit: int = Field(default=32, gt=0)
    result_limit: int = Field(default=8, gt=0)
