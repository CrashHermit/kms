"""Typed contracts for exercise discovery."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class ExerciseClassificationInput(BaseModel):
    """One independently contextualized exercise-classification target."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class ExerciseStartDecision(BaseModel):
    """Classification result for one local exercise-start position."""

    position: int = Field(ge=0, strict=True)
    is_exercise_start: bool


class ExerciseBoundaryDecision(BaseModel):
    """Classification result for one local exercise-boundary position."""

    position: int = Field(ge=0, strict=True)
    is_exercise_boundary: bool
