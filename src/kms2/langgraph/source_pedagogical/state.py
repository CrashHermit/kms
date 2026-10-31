"""LangGraph state for source-pedagogical preparation."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.exercise_splitter import SplitResult
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ExerciseComponent,
    PedagogicalComponent,
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionRequest,
)
from kms2.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionRequest,
)


class SourcePedagogicalState(BaseModel):
    """State shared by source-pedagogical stage nodes."""

    source_uuid: str
    source_pages: list[SourcePage] = Field(default_factory=list)
    split_pages: list[SourcePage] = Field(default_factory=list)
    embedded_pages: list[SourcePage] = Field(default_factory=list)
    split_results: Annotated[list[SplitResult], operator.add] = Field(
        default_factory=list
    )
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
    reembedded_block_count: int = 0
