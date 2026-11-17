"""Typed identities for schedulable learning targets and grounding."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PropositionTarget(BaseModel):
    """One proposition selected for assessment or review."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['proposition'] = 'proposition'
    uuid: str


class ArgumentTarget(BaseModel):
    """One argument selected for assessment or review."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['argument'] = 'argument'
    uuid: str


class SolutionTarget(BaseModel):
    """One task-specific reference solution selected for assessment or review."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['solution'] = 'solution'
    uuid: str


LearningTarget = Annotated[
    PropositionTarget | ArgumentTarget | SolutionTarget,
    Field(discriminator='kind'),
]


class GroundingReferences(BaseModel):
    """Owned knowledge and source units supporting generated material."""

    model_config = ConfigDict(extra='forbid')

    proposition_uuids: list[str] = Field(default_factory=list)
    argument_uuids: list[str] = Field(default_factory=list)
    unit_uuids: list[str] = Field(default_factory=list)
