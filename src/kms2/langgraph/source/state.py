"""LangGraph state for KMS2 source processing."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model import Source, SourcePage
from kms2.core.model.source_stage.content_correction import (
    ContentCorrectionResult,
)
from kms2.core.model.source_stage.embedding import EmbeddingResult
from kms2.core.model.source_stage.exercise_splitter import SplitResult
from kms2.core.model.source_stage.formatting import FormattingResult
from kms2.core.model.source_stage.image_description import (
    ImageDescriptionResult,
)
from kms2.core.model.source_stage.image_seam import ImageSeamResult
from kms2.core.model.source_stage.instruction import Instruction
from kms2.core.model.source_stage.pedagogical import (
    ExerciseComponent,
    PedagogicalComponent,
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_stage.text_seam import TextSeamResult


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

    image_description_results: Annotated[
        list[ImageDescriptionResult], operator.add
    ] = Field(default_factory=list)
    image_described_pages: list[SourcePage] = Field(default_factory=list)
    split_pages: list[SourcePage] = Field(default_factory=list)
    instructions: list[Instruction] = Field(default_factory=list)
    exercise_components: list[ExerciseComponent] = Field(default_factory=list)
    pedagogical_components: list[PedagogicalComponent] = Field(
        default_factory=list
    )
    statements: list[StatementDraft] = Field(default_factory=list)
    procedures: list[ProcedureDraft] = Field(default_factory=list)
    embedded_pages: list[SourcePage] = Field(default_factory=list)

    embedding_results: Annotated[list[EmbeddingResult], operator.add] = Field(
        default_factory=list
    )

    split_results: Annotated[list[SplitResult], operator.add] = Field(
        default_factory=list
    )
