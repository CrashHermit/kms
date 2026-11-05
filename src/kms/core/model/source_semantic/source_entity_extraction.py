"""Source entity extraction models."""

from pydantic import BaseModel, Field, model_validator

from kms.core.model.source_extraction import (
    SourceExtractionRequest,
)


class SourceEntityCandidate(BaseModel):
    """One explicitly discussed source-local entity mention."""

    name: str = Field(min_length=1)
    source_position: int = Field(ge=0, strict=True)


class SourceEntityExtractionResult(BaseModel):
    """Model outputs associated with one extraction request."""

    request: SourceExtractionRequest
    entities: list[SourceEntityCandidate]

    @model_validator(mode='after')
    def validate_source_positions(self) -> SourceEntityExtractionResult:
        """Reject output ownership outside this request's target blocks."""
        if any(
            candidate.source_position >= len(self.request.target.source_blocks)
            for candidate in self.entities
        ):
            raise ValueError(
                'entity source_position is outside the request target'
            )
        return self
