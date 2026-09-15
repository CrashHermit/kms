"""Fact extraction request and result models."""

from pydantic import BaseModel, Field

from kms2.core.model import SourceBlock
from kms2.core.model.context import (
    SourceBlockContext,
    SourceContextWindow,
)


class FactExtractionInput(BaseModel):
    """UUID-free context supplied to the fact extractor."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class FactExtractionRequest(BaseModel):
    """One canonical target block with its model-facing context."""

    source_uuid: str
    target_block: SourceBlock
    window: SourceContextWindow

    def model_input(self) -> FactExtractionInput:
        """Return the fact model's UUID-free canonical boundary."""
        return FactExtractionInput(
            context_before=self.window.context_before,
            target_block=self.window.target[0],
            context_after=self.window.context_after,
        )


class AtomicFact(BaseModel):
    """One source-faithful atomic assertion returned by the first pass."""

    text: str = Field(min_length=1)


class ExtractedFact(BaseModel):
    """One atomic fact reattached to its authoritative source block."""

    source_uuid: str
    source_block_uuid: str
    text: str = Field(min_length=1)


class FactExtractionResult(BaseModel):
    """Facts extracted from one source block."""

    target_block_uuid: str
    facts: list[ExtractedFact] = Field(default_factory=list)


__all__ = [
    'AtomicFact',
    'ExtractedFact',
    'FactExtractionInput',
    'FactExtractionRequest',
    'FactExtractionResult',
]
