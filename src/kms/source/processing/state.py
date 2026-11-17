"""LangGraph state for one complete source-processing run."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit
from kms.source.processing.correction.models import ContentCorrectionResult
from kms.source.processing.exercise_splitter.models import SplitResult
from kms.source.processing.formatting.models import FormattingResult
from kms.source.processing.image_filter.models import ImageFilterResult
from kms.source.processing.image_seam.models import ImageSeamResult
from kms.source.processing.instruction_redistribution.models import (
    ExerciseSupportGroup,
)
from kms.source.processing.procedure_splitter.models import ProcedureSplitResult
from kms.source.processing.text_seam.models import (
    TextSeamResult,
    TextSeamRewriteRequest,
    TextSeamRewriteResult,
)


class SourceProcessingState(BaseModel):
    """State shared by source processing and embedding."""

    input: PDFSourceInput
    source: Source

    @property
    def source_uuid(self) -> str:
        """Return the UUID of the source being processed."""
        return self.source.uuid

    selected_page_indexes: list[int] = Field(default_factory=list)
    ocr_page_images: dict[int, str] = Field(default_factory=dict)
    ocr_block_pages: dict[str, int] = Field(default_factory=dict)
    ocr_blocks: list[SourceBlock] = Field(default_factory=list)
    correction_results: Annotated[
        list[ContentCorrectionResult], operator.add
    ] = Field(default_factory=list)
    corrected_blocks: list[SourceBlock] = Field(default_factory=list)
    formatting_results: Annotated[list[FormattingResult], operator.add] = Field(
        default_factory=list
    )
    formatted_blocks: list[SourceBlock] = Field(default_factory=list)
    text_seam_results: Annotated[list[TextSeamResult], operator.add] = Field(
        default_factory=list
    )
    text_seam_rewrite_requests: list[TextSeamRewriteRequest] = Field(
        default_factory=list
    )
    text_seam_rewrite_results: Annotated[
        list[TextSeamRewriteResult], operator.add
    ] = Field(default_factory=list)
    text_seam_blocks: list[SourceBlock] = Field(default_factory=list)
    image_seam_results: Annotated[list[ImageSeamResult], operator.add] = Field(
        default_factory=list
    )
    image_seam_blocks: list[SourceBlock] = Field(default_factory=list)
    image_filter_results: Annotated[list[ImageFilterResult], operator.add] = (
        Field(default_factory=list)
    )
    image_filtered_blocks: list[SourceBlock] = Field(default_factory=list)
    exercise_split_results: Annotated[list[SplitResult], operator.add] = Field(
        default_factory=list
    )
    exercise_split_blocks: list[SourceBlock] = Field(default_factory=list)
    procedure_split_results: Annotated[
        list[ProcedureSplitResult], operator.add
    ] = Field(default_factory=list)
    procedure_split_blocks: list[SourceBlock] = Field(default_factory=list)
    exercise_support_groups: list[ExerciseSupportGroup] = Field(
        default_factory=list
    )
    exercise_units: list[SourceUnit] = Field(default_factory=list)
    procedure_units: list[SourceUnit] = Field(default_factory=list)
    units: list[SourceUnit] = Field(default_factory=list)
