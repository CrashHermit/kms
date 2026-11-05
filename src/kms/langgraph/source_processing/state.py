"""LangGraph state for KMS2 source processing."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.content_correction import (
    ContentCorrectionResult,
)
from kms.core.model.source_processing.formatting import FormattingResult
from kms.core.model.source_processing.image_description import (
    ImageDescriptionResult,
)
from kms.core.model.source_processing.image_filter import ImageFilterResult
from kms.core.model.source_processing.image_seam import ImageSeamResult
from kms.core.model.source_processing.input import PDFSourceInput
from kms.core.model.source_processing.text_seam import TextSeamResult


class SourceProcessingState(BaseModel):
    """State shared by the nodes that process one source."""

    input: PDFSourceInput
    source: Source

    @property
    def source_uuid(self) -> str:
        """Return the UUID of the source being processed."""
        return self.source.uuid

    ocr_pages: list[SourcePage] = Field(default_factory=list)
    ocr_page_images: dict[int, str] = Field(default_factory=dict)

    correction_results: Annotated[
        list[ContentCorrectionResult], operator.add
    ] = Field(default_factory=list)
    corrected_pages: list[SourcePage] = Field(default_factory=list)

    formatting_results: Annotated[list[FormattingResult], operator.add] = Field(
        default_factory=list
    )
    formatted_pages: list[SourcePage] = Field(default_factory=list)

    text_seam_even_results: Annotated[list[TextSeamResult], operator.add] = (
        Field(default_factory=list)
    )
    text_seam_even_pages: list[SourcePage] = Field(default_factory=list)
    text_seam_odd_results: Annotated[list[TextSeamResult], operator.add] = (
        Field(default_factory=list)
    )
    text_seam_pages: list[SourcePage] = Field(default_factory=list)

    image_seam_even_results: Annotated[list[ImageSeamResult], operator.add] = (
        Field(default_factory=list)
    )
    image_seam_even_pages: list[SourcePage] = Field(default_factory=list)
    image_seam_odd_results: Annotated[list[ImageSeamResult], operator.add] = (
        Field(default_factory=list)
    )
    image_seam_pages: list[SourcePage] = Field(default_factory=list)
    image_filter_results: Annotated[list[ImageFilterResult], operator.add] = (
        Field(default_factory=list)
    )
    image_filtered_pages: list[SourcePage] = Field(default_factory=list)

    image_description_results: Annotated[
        list[ImageDescriptionResult], operator.add
    ] = Field(default_factory=list)
    image_described_pages: list[SourcePage] = Field(default_factory=list)
    embedded_pages: list[SourcePage] = Field(default_factory=list)
