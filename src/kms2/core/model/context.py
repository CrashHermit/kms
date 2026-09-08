"""Model-facing source context records."""

from pydantic import BaseModel, Field

from kms2.core.model.block_types import BlockType


class SourceBlockContext(BaseModel):
    """Projected source-block data for downstream model inputs."""

    block_type: BlockType
    content: str | None = None
    asset_paths: list[str] = Field(default_factory=list)


class SourceContextWindow(BaseModel):
    """Ordered source context split around a target span."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target: list[SourceBlockContext] = Field(default_factory=list)
    context_after: list[SourceBlockContext] = Field(default_factory=list)
