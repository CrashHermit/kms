"""Source entity occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.source_extraction import (
    SourceEvidenceInput,
    SourceExtractionOccurrence,
)


class SourceEntity(SourceExtractionOccurrence):
    """Block-owned entity occurrence."""

    name: str = Field(min_length=1)


class SourceEntityDescriptionInput(SourceEvidenceInput):
    """Entity term and source-local description evidence."""

    term: str = Field(min_length=1)


class SourceEntityDescriptionItem(BaseModel):
    """One packet-local entity request with its own complete evidence."""

    position: int = Field(ge=0, strict=True)
    request: SourceEntityDescriptionInput


class SourceEntityDescriptionOutput(BaseModel):
    """Indexed description without source identities or evidence pointers."""

    position: int = Field(ge=0, strict=True)
    description: str


class SourceEntityDescriptionTarget(SourceEntity):
    """Raw entity occurrence selected for description generation."""


class SourceEntityDescriptionRequest(BaseModel):
    """One entity occurrence and its description input."""

    target: SourceEntityDescriptionTarget
    model_input: SourceEntityDescriptionInput


class SourceEntityDescriptionResult(SourceEntityDescriptionTarget):
    """One described entity occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None
