"""Typed contracts for pedagogical discovery."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class PedagogicalClassificationInput(BaseModel):
    """One independently contextualized pedagogical-classification target."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class PedagogicalStartDecision(BaseModel):
    """Classification result for one local pedagogical-start position."""

    position: int = Field(ge=0, strict=True)
    is_pedagogical_start: bool


class PedagogicalBoundaryDecision(BaseModel):
    """Classification result for one local pedagogical-boundary position."""

    position: int = Field(ge=0, strict=True)
    is_pedagogical_boundary: bool
