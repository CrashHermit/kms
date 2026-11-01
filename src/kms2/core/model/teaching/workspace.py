"""Private typed evidence registries and inspection workspace."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.block import SourceBlock
from kms2.core.model.knowledge.proposition import Proposition, PropositionRecord
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.source import Source
from kms2.core.model.source_evidence import (
    SourceEvidenceGroup,
    SourceProcedurePassage,
    SourceStatementPassage,
)
from kms2.core.model.source_processing.admission import (
    SourceAdmissionResult,
    StagedSourceSnapshot,
)
from kms2.core.model.source_semantic.source_procedure import SourceProcedure
from kms2.core.model.source_semantic.source_statement import SourceStatement
from kms2.core.model.teaching.acquisition import LocalEvidenceOutcome
from kms2.core.model.teaching.inference import TeachingToolObservation
from kms2.core.model.visual_asset import VisualAsset


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TeachingBlockRegistryEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    block: SourceBlock


class TeachingImageRegistryEntry(_StrictModel):
    asset: VisualAsset
    source: Source
    block_positions: list[int] = Field(default_factory=list)
    data: bytes | None = None


class TeachingGroupRegistryEntry(_StrictModel):
    source: Source
    group: SourceEvidenceGroup


class TeachingPropositionRegistryEntry(_StrictModel):
    uuid: str
    proposition: Proposition | None = None
    record: PropositionRecord | None = None


class TeachingReasoningRegistryEntry(_StrictModel):
    uuid: str
    record: ReasoningStepRecord | None = None


class TeachingStatementRegistryEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    statement: SourceStatement
    passage: SourceStatementPassage | None = None


class TeachingProcedureRegistryEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    procedure: SourceProcedure
    passage: SourceProcedurePassage | None = None


class BlockInspection(_StrictModel):
    kind: Literal['block'] = 'block'
    source: Source
    block: SourceBlock
    page_indexes: list[int]


class ImageInspection(_StrictModel):
    kind: Literal['image'] = 'image'
    source: Source
    asset: VisualAsset
    block_uuids: list[str]
    data: bytes


class PropositionInspection(_StrictModel):
    kind: Literal['proposition'] = 'proposition'
    record: PropositionRecord


InspectedEvidenceReceipt = Annotated[
    BlockInspection | ImageInspection | PropositionInspection,
    Field(discriminator='kind'),
]


class TeachingEvidenceWorkspace(_StrictModel):
    """Session-private registry, acquisition, and inspected evidence."""

    block_registry: list[TeachingBlockRegistryEntry] = Field(
        default_factory=list
    )
    image_registry: list[TeachingImageRegistryEntry] = Field(
        default_factory=list
    )
    group_registry: list[TeachingGroupRegistryEntry] = Field(
        default_factory=list
    )
    proposition_registry: list[TeachingPropositionRegistryEntry] = Field(
        default_factory=list
    )
    reasoning_registry: list[TeachingReasoningRegistryEntry] = Field(
        default_factory=list
    )
    statement_registry: list[TeachingStatementRegistryEntry] = Field(
        default_factory=list
    )
    procedure_registry: list[TeachingProcedureRegistryEntry] = Field(
        default_factory=list
    )
    inspected_block_positions: set[int] = Field(default_factory=set)
    inspected_image_positions: set[int] = Field(default_factory=set)
    inspected_proposition_positions: set[int] = Field(default_factory=set)
    inspected_reasoning_positions: set[int] = Field(default_factory=set)
    inspected_statement_positions: set[int] = Field(default_factory=set)
    inspected_procedure_positions: set[int] = Field(default_factory=set)
    staged_sources: list[StagedSourceSnapshot] = Field(default_factory=list)
    acquired_sources: list[SourceAdmissionResult] = Field(default_factory=list)
    local_outcomes: list[LocalEvidenceOutcome] = Field(default_factory=list)
    receipts: list[InspectedEvidenceReceipt] = Field(default_factory=list)
    observations: list[TeachingToolObservation] = Field(default_factory=list)
    active_block_positions: list[int] = Field(default_factory=list)
    active_image_positions: list[int] = Field(default_factory=list)
    active_proposition_positions: list[int] = Field(default_factory=list)
    active_reasoning_positions: list[int] = Field(default_factory=list)
    active_statement_positions: list[int] = Field(default_factory=list)
    active_procedure_positions: list[int] = Field(default_factory=list)
    discovered_block_positions: list[int] = Field(default_factory=list)
    discovered_image_positions: list[int] = Field(default_factory=list)
    discovered_proposition_positions: list[int] = Field(default_factory=list)
    discovered_reasoning_positions: list[int] = Field(default_factory=list)
    discovered_statement_positions: list[int] = Field(default_factory=list)
    discovered_procedure_positions: list[int] = Field(default_factory=list)
