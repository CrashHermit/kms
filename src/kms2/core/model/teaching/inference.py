"""Transient teaching context and typed actions for one prediction."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    TeachingLocalOutcome,
)
from kms2.core.model.teaching.observation import LearnerObservationKind
from kms2.core.model.teaching.operation import (
    RespondAction,
    TeachingAssessmentContext,
    TeachingPlanDraft,
)
from kms2.core.model.teaching.positions import Position
from kms2.core.model.teaching.question import (
    TeachingQuestionKind,
    TeachingResponseSpecification,
    TeachingResponseView,
)
from kms2.core.model.teaching.session import TeachingPhase
from kms2.core.model.teaching.turn import TeachingRole


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TeachingQuestionContext(_StrictModel):
    """Current-session question projection with backend-derived step linkage."""

    kind: TeachingQuestionKind
    prompt: str
    response: TeachingResponseSpecification
    step_position: Position | None


class SearchSourcesAction(_StrictModel):
    """Search the user's eligible local source library."""

    kind: Literal['search_sources']
    query: str


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


class ReadPropositionsAction(_StrictModel):
    kind: Literal['read_propositions']
    proposition_positions: list[Position]


class ReadReasoningStepsAction(_StrictModel):
    kind: Literal['read_reasoning_steps']
    reasoning_positions: list[Position]


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
    reasoning_positions: list[Position] = Field(default_factory=list)
    statement_positions: list[Position] = Field(default_factory=list)
    procedure_positions: list[Position] = Field(default_factory=list)


class ListAcquiredBlocksAction(_StrictModel):
    """List one bounded block-catalog page from an acquired source."""

    kind: Literal['list_acquired_blocks'] = 'list_acquired_blocks'
    source_position: Position
    start: Position


EvidenceToolAction = Annotated[
    SearchSourcesAction
    | ReadPassagesAction
    | ReadImagesAction
    | SearchPropositionsAction
    | ReadPropositionsAction
    | ReadReasoningStepsAction
    | SearchStatementsAction
    | ReadStatementsAction
    | SearchProceduresAction
    | ReadProceduresAction
    | FocusEvidenceAction
    | ListAcquiredBlocksAction
    | AcquireEvidenceAction,
    Field(discriminator='kind'),
]

TeachingAction = Annotated[
    SearchSourcesAction
    | ReadPassagesAction
    | ReadImagesAction
    | SearchPropositionsAction
    | ReadPropositionsAction
    | ReadReasoningStepsAction
    | SearchStatementsAction
    | ReadStatementsAction
    | SearchProceduresAction
    | ReadProceduresAction
    | FocusEvidenceAction
    | ListAcquiredBlocksAction
    | AcquireEvidenceAction
    | RespondAction,
    Field(discriminator='kind'),
]


class TeachingContextTurn(_StrictModel):
    position: Position
    role: TeachingRole
    content: str
    question: TeachingQuestionContext | None
    responds_to_pending_question: bool
    presents_current_plan: bool
    accepts_current_plan: bool


class TeachingContextBlock(_StrictModel):
    position: Position
    source_key: str
    page_indexes: list[int]
    block_type: str
    content: str | None
    image_positions: list[Position]
    inspected: bool


class TeachingContextImage(_StrictModel):
    position: Position
    block_positions: list[Position]
    inspected: bool
    native_image_position: Position | None


class TeachingContextGroup(_StrictModel):
    position: Position
    kind: Literal['statement', 'procedure', 'instruction']
    member_block_positions: list[Position]
    governed_statement_positions: list[Position]


class TeachingContextProposition(_StrictModel):
    position: Position
    content: str
    inspected: bool
    supporting_block_positions: list[Position] = Field(default_factory=list)
    reasoning_positions: list[Position] = Field(default_factory=list)


class TeachingContextReasoningStep(_StrictModel):
    position: Position
    explanation: str | None
    premise_positions: list[Position]
    conclusion_position: Position | None
    supporting_block_positions: list[Position]
    inspected: bool


class TeachingContextStatement(_StrictModel):
    position: Position
    content: str
    is_exercise: bool
    block_positions: list[Position]
    inspected: bool


class TeachingContextProcedure(_StrictModel):
    position: Position
    content: str
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


class TeachingReasoningDiscovery(_StrictModel):
    position: Position


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
    reasoning_steps: list[TeachingReasoningDiscovery] = Field(
        default_factory=list
    )
    statements: list[TeachingStatementDiscovery] = Field(default_factory=list)
    procedures: list[TeachingProcedureDiscovery] = Field(default_factory=list)


class TeachingAcquiredSource(_StrictModel):
    position: Position
    source_key: str
    page_count: int
    block_count: int
    image_count: int


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
        'list_acquired_blocks',
        'search_propositions',
        'read_propositions',
        'read_reasoning_steps',
        'search_statements',
        'read_statements',
        'search_procedures',
        'read_procedures',
    ]
    query: str | None
    block_positions: list[Position]
    image_positions: list[Position]
    groups: list[TeachingContextGroup]
    proposition_positions: list[Position] = Field(default_factory=list)
    reasoning_positions: list[Position] = Field(default_factory=list)
    statement_positions: list[Position] = Field(default_factory=list)
    procedure_positions: list[Position] = Field(default_factory=list)
    local_outcome_position: Position | None = None
    acquired_source_positions: list[Position] = Field(default_factory=list)
    catalog_start: Position | None = None


class TeachingContext(_StrictModel):
    """Bounded state and inspected evidence for one prediction."""

    goal: str
    phase: TeachingPhase
    current_assessment: TeachingAssessmentContext | None
    current_plan: TeachingPlanDraft | None
    current_plan_accepted: bool
    current_step_position: Position | None
    pending_question: TeachingQuestionContext | None
    turns: list[TeachingContextTurn]
    blocks: list[TeachingContextBlock]
    learner_history: list[TeachingLearnerObservation] = Field(
        default_factory=list
    )
    learner_history_truncated: bool = False
    images: list[TeachingContextImage]
    propositions: list[TeachingContextProposition] = Field(default_factory=list)
    image_limit: int = Field(default=4, ge=0)
    reasoning_steps: list[TeachingContextReasoningStep] = Field(
        default_factory=list
    )
    statements: list[TeachingContextStatement] = Field(default_factory=list)
    procedures: list[TeachingContextProcedure] = Field(default_factory=list)
    observations: list[TeachingToolObservation]
    tools_remaining: int
    history_truncated: bool
    research_enabled: bool = False
    discovery: TeachingDiscoveryCatalog = Field(
        default_factory=TeachingDiscoveryCatalog
    )
    acquired_sources: list[TeachingAcquiredSource] = Field(default_factory=list)
    citation_candidates: list[TeachingCitationCandidate] = Field(
        default_factory=list
    )
    local_outcomes: list[TeachingLocalOutcome] = Field(default_factory=list)
