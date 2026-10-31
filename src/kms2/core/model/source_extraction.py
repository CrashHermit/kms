"""Shared source-evidence contracts for independent extraction streams."""

from pydantic import BaseModel, Field, model_validator

from kms2.core.model.base import Vertex
from kms2.core.model.block import SourceBlock
from kms2.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms2.core.windowing import project_block


class SourceExtractionTarget(BaseModel):
    """Ordered owning evidence for one extraction request."""

    source_blocks: list[SourceBlock] = Field(min_length=1)


class SourceExtractionContext(BaseModel):
    """Ordered reference-only evidence for one extraction request."""

    source_blocks: list[SourceBlock] = Field(default_factory=list)


class SourceEvidenceInput(BaseModel):
    """UUID-free owning evidence and reference context for downstream inference."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_blocks: list[SourceBlockContext] = Field(min_length=1, max_length=1)
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceExtractionInput(BaseModel):
    """Indexed target blocks and reference-only context for extraction."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_blocks: list[PositionedSourceBlock] = Field(min_length=1)
    context_after: list[SourceBlockContext] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_positions(self) -> SourceExtractionInput:
        """Keep target positions contiguous and local to this request."""
        if any(
            block.position != position
            for position, block in enumerate(self.target_blocks)
        ):
            raise ValueError(
                'target positions must be contiguous and zero-based'
            )
        return self


class SourceExtractionRequest(BaseModel):
    """In-memory extraction job scope with UUID-free evidence references."""

    source_uuid: str
    target: SourceExtractionTarget
    context_before: SourceExtractionContext
    context_after: SourceExtractionContext

    def model_input(self) -> SourceExtractionInput:
        """Project indexed targets and unindexed context without graph identities."""
        return SourceExtractionInput(
            context_before=[
                project_block(block)
                for block in self.context_before.source_blocks
            ],
            target_blocks=[
                PositionedSourceBlock(
                    position=position,
                    source_block=project_block(block),
                )
                for position, block in enumerate(self.target.source_blocks)
            ],
            context_after=[
                project_block(block)
                for block in self.context_after.source_blocks
            ],
        )


class SourceExtractionOccurrence(Vertex):
    """One source output with one block-derived evidence owner."""

    source_uuid: str
    target: SourceExtractionTarget

    @model_validator(mode='after')
    def validate_owner(self) -> SourceExtractionOccurrence:
        """Require exactly one owning block."""
        if len(self.target.source_blocks) != 1:
            raise ValueError(
                'an extraction occurrence must own exactly one source block'
            )
        return self
