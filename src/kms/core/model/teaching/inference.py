"""Transient teaching context and typed actions for one prediction."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.material import (
    MaterialContext,
    SourceGroupView,
)
from kms.core.model.source import SourceRevisionReference
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    TeachingLocalOutcome,
)
from kms.core.model.teaching.observation import LearnerObservationKind
from kms.core.model.teaching.operation import (
    RespondAction,
    TeachingAssessmentContext,
    TeachingPlanDraft,
)
from kms.core.model.teaching.phase import TeachingPhase
from kms.core.model.teaching.positions import Position
from kms.core.model.teaching.question import (
    TeachingQuestionContext,
    TeachingQuestionKind,
    TeachingResponseView,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SearchSourcesAction(_StrictModel):
    """Search source blocks, optionally limited to explicitly offered sources."""

    kind: Literal['search_sources']
    query: str
    source_uuids: list[str] | None = None


class ReadPassagesAction(_StrictModel):
    """Read complete source blocks and optional whole-block context."""

    kind: Literal['read_passages']
    block_positions: list[Position]
    expand_memberships: bool
    before: Position
    after: Position


class ReadImagesAction(_StrictModel):
    """Read original image assets discovered through inspected passages."""

    kind: Literal['read_images']
    image_positions: list[Position]


class SearchPropositionsAction(_StrictModel):
    kind: Literal['search_propositions']
    query: str


class SearchArgumentsAction(_StrictModel):
    kind: Literal['search_arguments']
    query: str


class SearchLearnerObservationsAction(_StrictModel):
    kind: Literal['search_learner_observations']
    query: str


class ReadLearnerObservationsAction(_StrictModel):
    kind: Literal['read_learner_observations']
    observation_positions: list[Position]


class ReadPropositionsAction(_StrictModel):
    kind: Literal['read_propositions']
    proposition_positions: list[Position]


class ReadArgumentsAction(_StrictModel):
    kind: Literal['read_arguments']
    argument_positions: list[Position]


class SearchStatementsAction(_StrictModel):
    kind: Literal['search_statements']
    query: str
    is_exercise: bool | None = None


class ReadStatementsAction(_StrictModel):
    kind: Literal['read_statements']
    statement_positions: list[Position]


class SearchProceduresAction(_StrictModel):
    kind: Literal['search_procedures']
    query: str


class ReadProceduresAction(_StrictModel):
    kind: Literal['read_procedures']
    procedure_positions: list[Position]


class FocusEvidenceAction(_StrictModel):
    """Replace the active inspected evidence packet by stable positions."""

    kind: Literal['focus_evidence'] = 'focus_evidence'
    block_positions: list[Position] = Field(default_factory=list)
    image_positions: list[Position] = Field(default_factory=list)
    proposition_positions: list[Position] = Field(default_factory=list)
    argument_positions: list[Position] = Field(default_factory=list)
    statement_positions: list[Position] = Field(default_factory=list)
    procedure_positions: list[Position] = Field(default_factory=list)


EvidenceToolAction = Annotated[
    SearchSourcesAction
    | ReadPassagesAction
    | ReadImagesAction
    | SearchPropositionsAction
    | SearchArgumentsAction
    | SearchLearnerObservationsAction
    | ReadLearnerObservationsAction
    | ReadArgumentsAction
    | SearchStatementsAction
    | ReadStatementsAction
    | SearchProceduresAction
    | ReadProceduresAction
    | FocusEvidenceAction
    | AcquireEvidenceAction,
    Field(discriminator='kind'),
]

TeachingAction = Annotated[
    SearchSourcesAction
    | ReadPassagesAction
    | ReadImagesAction
    | SearchPropositionsAction
    | SearchArgumentsAction
    | SearchLearnerObservationsAction
    | ReadLearnerObservationsAction
    | ReadArgumentsAction
    | SearchStatementsAction
    | ReadStatementsAction
    | SearchProceduresAction
    | ReadProceduresAction
    | FocusEvidenceAction
    | AcquireEvidenceAction
    | RespondAction,
    Field(discriminator='kind'),
]


class TeachingContextImage(_StrictModel):
    position: Position
    block_positions: list[Position]
    inspected: bool


class TeachingBlockDiscovery(_StrictModel):
    position: Position
    source_key: str
    page_indexes: list[int]
    block_type: str
    preview: str | None


class TeachingImageDiscovery(_StrictModel):
    position: Position
    block_positions: list[Position]


class TeachingPropositionDiscovery(_StrictModel):
    position: Position
    preview: str


class TeachingArgumentDiscovery(_StrictModel):
    position: Position
    preview: str | None = None


class TeachingLearnerObservationDiscovery(_StrictModel):
    position: Position
    kind: LearnerObservationKind
    topic: str
    preview: str


class TeachingStatementDiscovery(_StrictModel):
    position: Position
    source_key: str
    is_exercise: bool
    preview: str


class TeachingProcedureDiscovery(_StrictModel):
    position: Position
    source_key: str
    preview: str


class TeachingDiscoveryCatalog(_StrictModel):
    blocks: list[TeachingBlockDiscovery] = Field(default_factory=list)
    images: list[TeachingImageDiscovery] = Field(default_factory=list)
    propositions: list[TeachingPropositionDiscovery] = Field(
        default_factory=list
    )
    arguments: list[TeachingArgumentDiscovery] = Field(default_factory=list)
    statements: list[TeachingStatementDiscovery] = Field(default_factory=list)
    procedures: list[TeachingProcedureDiscovery] = Field(default_factory=list)
    learner_observations: list[TeachingLearnerObservationDiscovery] = Field(
        default_factory=list
    )


class TeachingLearnerQuestion(_StrictModel):
    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseView


class TeachingLearnerEvidence(_StrictModel):
    content: str
    created_at: datetime
    question: TeachingLearnerQuestion | None


class TeachingLearnerObservation(_StrictModel):
    position: Position
    registry_position: Position
    kind: LearnerObservationKind
    topic: str
    description: str
    created_at: datetime
    evidence: list[TeachingLearnerEvidence]


class TeachingCitationCandidate(_StrictModel):
    position: Position
    kind: Literal['block', 'image', 'proposition']
    evidence_position: Position


class TeachingToolObservation(_StrictModel):
    kind: Literal[
        'search_sources',
        'read_passages',
        'read_images',
        'acquire_local',
        'acquire_external',
        'focus_evidence',
        'search_propositions',
        'read_propositions',
        'read_arguments',
        'search_arguments',
        'search_learner_observations',
        'read_learner_observations',
        'search_statements',
        'read_statements',
        'search_procedures',
        'read_procedures',
        'invalid_action',
    ]
    query: str | None
    block_positions: list[Position]
    image_positions: list[Position]
    groups: list[SourceGroupView]
    proposition_positions: list[Position] = Field(default_factory=list)
    argument_positions: list[Position] = Field(default_factory=list)
    statement_positions: list[Position] = Field(default_factory=list)
    procedure_positions: list[Position] = Field(default_factory=list)
    learner_observation_positions: list[Position] = Field(default_factory=list)
    error: str | None = None
    local_outcome_position: Position | None = None
    source_revisions: list[SourceRevisionReference] = Field(
        default_factory=list
    )


class TeachingContext(_StrictModel):
    """Bounded state and inspected evidence for one prediction."""

    goal: str
    phase: TeachingPhase
    current_assessment: TeachingAssessmentContext | None
    current_plan: TeachingPlanDraft | None
    current_plan_accepted: bool
    current_step_position: Position | None
    pending_question: TeachingQuestionContext | None
    learner_history: list[TeachingLearnerObservation] = Field(
        default_factory=list
    )
    learner_history_truncated: bool = False
    material: MaterialContext = Field(default_factory=MaterialContext)
    images: list[TeachingContextImage] = Field(default_factory=list)
    groups: list[SourceGroupView] = Field(default_factory=list)
    image_limit: int = Field(default=4, ge=0)
    observations: list[TeachingToolObservation]
    tools_remaining: int
    history_truncated: bool
    research_enabled: bool = False
    discovery: TeachingDiscoveryCatalog = Field(
        default_factory=TeachingDiscoveryCatalog
    )
    citation_candidates: list[TeachingCitationCandidate] = Field(
        default_factory=list
    )
    local_outcomes: list[TeachingLocalOutcome] = Field(default_factory=list)
