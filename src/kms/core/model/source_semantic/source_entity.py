"""Source entity occurrence and description models."""

from pydantic import BaseModel, Field

from kms.core.content import MultimodalContentBuilder
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_extraction import (
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
    """One entity occurrence and its canonical description window."""

    target: SourceEntityDescriptionTarget
    window: SourceBlockWindow

    def model_input(
        self, builder: MultimodalContentBuilder
    ) -> SourceEntityDescriptionInput:
        """Project occurrence evidence into the current inference catalog."""
        evidence = SourceEvidenceInput.from_window(self.window, builder)
        return SourceEntityDescriptionInput(
            term=self.target.name,
            context_before=evidence.context_before,
            target_blocks=evidence.target_blocks,
            context_after=evidence.context_after,
        )


class SourceEntityDescriptionResult(SourceEntityDescriptionTarget):
    """One described entity occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None
