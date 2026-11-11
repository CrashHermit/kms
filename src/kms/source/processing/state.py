"""LangGraph state for one complete source-processing run."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms.source.models.inputs import PDFSourceInput
from kms.source.models.material import (
    ExerciseComponent,
    Instruction,
    PedagogicalComponent,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.source import Source, SourcePage
from kms.source.processing.correction.models import ContentCorrectionResult
from kms.source.processing.exercise_splitter.models import SplitResult
from kms.source.processing.formatting.models import FormattingResult
from kms.source.processing.image_filter.models import ImageFilterResult
from kms.source.processing.image_seam.models import (
    ImageSeamRequest,
    ImageSeamResult,
)
from kms.source.processing.procedure_reconstruction.models import (
    SourceProcedureReconstructionRequest,
)
from kms.source.processing.statement_reconstruction.models import (
    SourceStatementReconstructionRequest,
)
from kms.source.processing.text_seam.models import TextSeamResult


class SourceProcessingState(BaseModel):
    """State shared by source capture, pedagogical extraction, and embedding."""

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
    image_seam_requests: list[ImageSeamRequest] = Field(default_factory=list)
    image_seam_results: Annotated[list[ImageSeamResult], operator.add] = Field(
        default_factory=list
    )
    image_seam_pages: list[SourcePage] = Field(default_factory=list)
    image_filter_results: Annotated[list[ImageFilterResult], operator.add] = (
        Field(default_factory=list)
    )
    image_filtered_pages: list[SourcePage] = Field(default_factory=list)
    split_results: Annotated[list[SplitResult], operator.add] = Field(
        default_factory=list
    )
    split_pages: list[SourcePage] = Field(default_factory=list)
    instructions: list[Instruction] = Field(default_factory=list)
    exercise_components: list[ExerciseComponent] = Field(default_factory=list)
    pedagogical_components: list[PedagogicalComponent] = Field(
        default_factory=list
    )
    statements: list[StatementDraft] = Field(default_factory=list)
    procedures: list[ProcedureDraft] = Field(default_factory=list)
    source_statement_reconstruction_requests: list[
        SourceStatementReconstructionRequest
    ] = Field(default_factory=list)
    source_procedure_reconstruction_requests: list[
        SourceProcedureReconstructionRequest
    ] = Field(default_factory=list)
    source_statement_reconstruction_results: Annotated[
        list[StatementDraft], operator.add
    ] = Field(default_factory=list)
    source_procedure_reconstruction_results: Annotated[
        list[ProcedureDraft], operator.add
    ] = Field(default_factory=list)
