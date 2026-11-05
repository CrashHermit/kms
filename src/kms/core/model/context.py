"""Model-facing source context records and retained source windows."""

from pydantic import BaseModel, Field

from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent


class SourceBlockContext(BaseModel):
    """Projected source-block data for downstream model inputs."""

    block_type: BlockType
    material: MultimodalContent


class PositionedSourceBlock(BaseModel):
    """One source block with a zero-based position in a request target."""

    position: int = Field(ge=0, strict=True)
    source_block: SourceBlockContext


class SourceBlockWindow(BaseModel):
    """Selected canonical source blocks retained until model projection."""

    context_before: list[SourceBlock] = Field(default_factory=list)
    target: list[SourceBlock] = Field(default_factory=list)
    context_after: list[SourceBlock] = Field(default_factory=list)


class SourceContextWindow(BaseModel):
    """Ordered model-facing source context split around a target span."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target: list[SourceBlockContext] = Field(default_factory=list)
    context_after: list[SourceBlockContext] = Field(default_factory=list)
