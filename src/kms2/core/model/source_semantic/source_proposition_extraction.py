"""Source proposition extraction and embedding models."""

from pydantic import BaseModel, Field, model_validator

from kms2.core.model.source_extraction import (
    SourceExtractionOccurrence,
    SourceExtractionRequest,
)


class SourcePropositionCandidate(BaseModel):
    """One independently meaningful claim extracted from target evidence."""

    text: str = Field(min_length=1)
    source_position: int = Field(ge=0, strict=True)


class SourceProposition(SourceExtractionOccurrence):
    """Block-owned proposition occurrence."""

    text: str = Field(min_length=1)


class SourcePropositionExtractionResult(BaseModel):
    """Model outputs associated with one extraction request."""

    request: SourceExtractionRequest
    propositions: list[SourcePropositionCandidate]

    @model_validator(mode='after')
    def validate_source_positions(self) -> SourcePropositionExtractionResult:
        """Reject output ownership outside this request's target blocks."""
        if any(
            candidate.source_position >= len(self.request.target.source_blocks)
            for candidate in self.propositions
        ):
            raise ValueError(
                'proposition source_position is outside the request target'
            )
        return self


class SourcePropositionEmbeddingResult(SourceProposition):
    """A proposition occurrence with its complete-text embedding."""

    embedding: list[float] = Field(min_length=1)
