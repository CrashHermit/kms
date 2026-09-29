"""Fact extraction request and result models."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.block import SourceBlock
from kms2.core.model.context import SourceBlockContext
from kms2.core.windowing import project_block


class SourceFactTarget(Vertex):
    """Persisted source-block span that bears one fact's evidence."""

    source_blocks: list[SourceBlock] = Field(default_factory=list)


class SourceFactContext(Vertex):
    """Persisted source-block span used only to resolve references."""

    source_blocks: list[SourceBlock] = Field(default_factory=list)


class SourceFactExtractionInput(BaseModel):
    """UUID-free context supplied to the fact extractor."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_blocks: list[SourceBlockContext] = Field(default_factory=list)
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceFactExtractionRequest(BaseModel):
    """One persisted fact-evidence selection with its model-facing context."""

    source_uuid: str
    target: SourceFactTarget
    context_before: SourceFactContext
    context_after: SourceFactContext

    def model_input(self) -> SourceFactExtractionInput:
        """Return the UUID-free model boundary projected from graph pointers."""
        return SourceFactExtractionInput(
            context_before=[
                project_block(block)
                for block in self.context_before.source_blocks
            ],
            target_blocks=[
                project_block(block) for block in self.target.source_blocks
            ],
            context_after=[
                project_block(block)
                for block in self.context_after.source_blocks
            ],
        )


class SourceAtomicFact(BaseModel):
    """One source-faithful atomic fact returned by the first pass."""

    text: str = Field(min_length=1)


class SourceFact(Vertex):
    """Durable source-scoped atomic fact with node-first evidence pointers."""

    text: str = Field(min_length=1)
    target: SourceFactTarget
    context_before: SourceFactContext
    context_after: SourceFactContext


class SourceFactExtractionResult(BaseModel):
    """Facts extracted from one persisted target and its contexts."""

    target: SourceFactTarget
    context_before: SourceFactContext
    context_after: SourceFactContext
    facts: list[SourceAtomicFact] = Field(default_factory=list)
