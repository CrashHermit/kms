"""Neutral model-facing projections of shared workflow material."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.content import MultimodalContent
from kms.core.model.teaching.positions import Position
from kms.core.model.teaching.question import TeachingQuestionContext
from kms.core.model.teaching.turn import TeachingRole


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceBlockMaterial(_StrictModel):
    position: Position
    source_key: str
    page_indexes: list[int]
    block_type: str
    material: MultimodalContent
    asset_positions: list[Position]
    inspected: bool


class PropositionMaterial(_StrictModel):
    position: Position
    material: MultimodalContent
    inspected: bool
    supporting_block_positions: list[Position] = Field(default_factory=list)
    argument_positions: list[Position] = Field(default_factory=list)


class ArgumentMaterial(_StrictModel):
    position: Position
    material: MultimodalContent
    premise_positions: list[Position]
    conclusion_position: Position | None
    supporting_block_positions: list[Position]
    inspected: bool


class TurnMaterial(_StrictModel):
    position: Position
    role: TeachingRole
    material: MultimodalContent
    question: TeachingQuestionContext | None
    responds_to_pending_question: bool
    presents_current_plan: bool
    accepts_current_plan: bool


class MaterialContext(_StrictModel):
    blocks: list[SourceBlockMaterial] = Field(default_factory=list)
    propositions: list[PropositionMaterial] = Field(default_factory=list)
    arguments: list[ArgumentMaterial] = Field(default_factory=list)
    turns: list[TurnMaterial] = Field(default_factory=list)


class SourceGroupView(_StrictModel):
    position: Position
    kind: Literal['statement', 'procedure', 'instruction']
    member_block_positions: list[Position]
    governed_statement_positions: list[Position]
    is_exercise: bool | None = None
    material: MultimodalContent | None = None
