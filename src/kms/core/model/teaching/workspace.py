"""Private typed evidence registries and inspection workspace."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.block import SourceBlock
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import Proposition, PropositionRecord
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceEvidenceGroup,
    SourceProcedurePassage,
    SourceStatementPassage,
)
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.source_workspace import SourceWorkspace
from kms.core.model.teaching.acquisition import LocalEvidenceOutcome
from kms.core.model.teaching.inference import TeachingToolObservation
from kms.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationRecord,
)
from kms.core.model.visual_asset import VisualAsset


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TeachingGroupRegistryEntry(_StrictModel):
    source: Source
    group: SourceEvidenceGroup


class TeachingPropositionRegistryEntry(_StrictModel):
    uuid: str
    proposition: Proposition | None = None
    record: PropositionRecord | None = None


class TeachingArgumentRegistryEntry(_StrictModel):
    uuid: str
    preview: str | None = None
    record: ArgumentRecord | None = None


class TeachingLearnerObservationRegistryEntry(_StrictModel):
    observation: LearnerObservation
    record: LearnerObservationRecord | None = None


class SavedTeachingLearnerObservationRegistryEntry(_StrictModel):
    uuid: str
    was_read: bool


class _StatementEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    statement: SourceStatement


class TeachingStatementRegistryEntry(_StatementEntry):
    passage: SourceStatementPassage | None = None


class SavedTeachingStatementRegistryEntry(_StatementEntry):
    was_read: bool


class _ProcedureEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    procedure: SourceProcedure


class TeachingProcedureRegistryEntry(_ProcedureEntry):
    passage: SourceProcedurePassage | None = None


class SavedTeachingProcedureRegistryEntry(_ProcedureEntry):
    was_read: bool


class SavedSourceBlockEntry(_StrictModel):
    source: Source
    page_indexes: list[int]
    block_uuid: str


class SavedSourceImageEntry(_StrictModel):
    source: Source
    asset_uuid: str
    block_positions: list[int]


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


class SavedBlockInspection(_StrictModel):
    kind: Literal['block'] = 'block'
    source: Source
    block_uuid: str
    page_indexes: list[int]


class SavedImageInspection(_StrictModel):
    kind: Literal['image'] = 'image'
    source: Source
    asset_uuid: str
    block_uuids: list[str]


class PropositionInspection(_StrictModel):
    kind: Literal['proposition'] = 'proposition'
    record: PropositionRecord


InspectedEvidenceReceipt = Annotated[
    BlockInspection | ImageInspection | PropositionInspection,
    Field(discriminator='kind'),
]

SavedInspectedEvidenceReceipt = Annotated[
    SavedBlockInspection | SavedImageInspection | PropositionInspection,
    Field(discriminator='kind'),
]


class _TeachingEvidenceWorkspaceFields(_StrictModel):
    group_registry: list[TeachingGroupRegistryEntry] = Field(
        default_factory=list
    )
    proposition_registry: list[TeachingPropositionRegistryEntry] = Field(
        default_factory=list
    )
    argument_registry: list[TeachingArgumentRegistryEntry] = Field(
        default_factory=list
    )
    read_block_positions: set[int] = Field(default_factory=set)
    read_image_positions: set[int] = Field(default_factory=set)
    inspected_block_positions: set[int] = Field(default_factory=set)
    inspected_image_positions: set[int] = Field(default_factory=set)
    inspected_proposition_positions: set[int] = Field(default_factory=set)
    inspected_argument_positions: set[int] = Field(default_factory=set)
    inspected_statement_positions: set[int] = Field(default_factory=set)
    inspected_procedure_positions: set[int] = Field(default_factory=set)
    local_outcomes: list[LocalEvidenceOutcome] = Field(default_factory=list)
    observations: list[TeachingToolObservation] = Field(default_factory=list)
    active_block_positions: list[int] = Field(default_factory=list)
    active_image_positions: list[int] = Field(default_factory=list)
    active_proposition_positions: list[int] = Field(default_factory=list)
    active_argument_positions: list[int] = Field(default_factory=list)
    active_statement_positions: list[int] = Field(default_factory=list)
    active_procedure_positions: list[int] = Field(default_factory=list)
    discovered_block_positions: list[int] = Field(default_factory=list)
    discovered_image_positions: list[int] = Field(default_factory=list)
    discovered_proposition_positions: list[int] = Field(default_factory=list)
    discovered_argument_positions: list[int] = Field(default_factory=list)
    active_learner_observation_positions: list[int] = Field(
        default_factory=list
    )
    discovered_statement_positions: list[int] = Field(default_factory=list)
    discovered_procedure_positions: list[int] = Field(default_factory=list)
    discovered_learner_observation_positions: list[int] = Field(
        default_factory=list
    )


class TeachingEvidenceWorkspace(
    SourceWorkspace, _TeachingEvidenceWorkspaceFields
):
    """Session-private registry and inspected evidence."""

    statement_registry: list[TeachingStatementRegistryEntry] = Field(
        default_factory=list
    )
    procedure_registry: list[TeachingProcedureRegistryEntry] = Field(
        default_factory=list
    )
    receipts: list[InspectedEvidenceReceipt] = Field(default_factory=list)
    learner_observation_registry: list[
        TeachingLearnerObservationRegistryEntry
    ] = Field(default_factory=list)


class SavedTeachingEvidenceWorkspace(_TeachingEvidenceWorkspaceFields):
    """Reference-only teaching workspace stored with a session."""

    block_registry: list[SavedSourceBlockEntry] = Field(default_factory=list)
    image_registry: list[SavedSourceImageEntry] = Field(default_factory=list)
    statement_registry: list[SavedTeachingStatementRegistryEntry] = Field(
        default_factory=list
    )
    procedure_registry: list[SavedTeachingProcedureRegistryEntry] = Field(
        default_factory=list
    )
    learner_observation_registry: list[
        SavedTeachingLearnerObservationRegistryEntry
    ] = Field(default_factory=list)
    receipts: list[SavedInspectedEvidenceReceipt] = Field(default_factory=list)
