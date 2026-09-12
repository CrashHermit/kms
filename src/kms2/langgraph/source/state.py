"""LangGraph state for KMS2 source processing."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.content_correction import ContentCorrectionResult
from kms2.core.model.embedding import EmbeddingResult
from kms2.core.model.formatting import FormattingResult
from kms2.core.model.image_enrichment import ImageEnrichmentResult
from kms2.core.model.image_seam import ImageSeamResult
from kms2.core.model.source import Source, SourcePage
from kms2.core.model.splitter import SplitResult
from kms2.core.model.text_seam import TextSeamResult


class SourceState(BaseModel):
    """State shared by the nodes that process one source."""

    pdf_path: str
    pages: list[int] | None = Field(default=None)
    source: Source
    ocr_pages: list[SourcePage] = Field(default_factory=list)

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

    image_enrichment_results: Annotated[
        list[ImageEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    image_enriched_pages: list[SourcePage] = Field(default_factory=list)
    split_pages: list[SourcePage] = Field(default_factory=list)
    embedded_pages: list[SourcePage] = Field(default_factory=list)

    embedding_results: Annotated[list[EmbeddingResult], operator.add] = Field(
        default_factory=list
    )

    split_results: Annotated[list[SplitResult], operator.add] = Field(
        default_factory=list
    )
