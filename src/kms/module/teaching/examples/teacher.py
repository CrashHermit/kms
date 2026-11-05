"""Explicit runtime demonstrations for goal-driven teaching."""

import dspy

from kms.core.model.content import MultimodalContent
from kms.core.model.material import (
    MaterialContext,
    SourceBlockMaterial,
    TurnMaterial,
)
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
    LocalEvidenceRequest,
    TeachingLocalOutcome,
)
from kms.core.model.teaching.inference import (
    ReadProceduresAction,
    RespondAction,
    SearchProceduresAction,
    TeachingCitationCandidate,
    TeachingContext,
    TeachingDiscoveryCatalog,
    TeachingProcedureDiscovery,
    TeachingQuestionContext,
)
from kms.core.model.teaching.observation import (
    LearnerObservationDraft,
    LearnerObservationKind,
)
from kms.core.model.teaching.operation import (
    Consolidate,
    Hint,
    RepairCheck,
    TeachingAssessmentContext,
    TeachingAssessmentDraft,
    TeachingPlanDraft,
    TeachingPlanStepDraft,
    TeachingQuestionDraft,
    TeachingResponseSegment,
    TeachStep,
)
from kms.core.model.teaching.probe import ProbeStrand
from kms.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms.core.model.teaching.session import TeachingPhase
from kms.core.model.teaching.turn import TeachingRole


def _assessment() -> TeachingAssessmentDraft:
    """Build an assessment for an unestablished concept."""
    return TeachingAssessmentDraft(
        summary='Matrix multiplication is recognized; scalar geometry is not yet established.',
        strands=[
            ProbeStrand(
                topic='Eigenvectors',
                demonstrated_understanding='Recognizes matrix-vector multiplication.',
                observed_limit='Has not established how scalar multiplication changes length or orientation.',
                gaps=[
                    'How positive, negative, and zero scalars affect a nonzero vector.'
                ],
                misconceptions=[],
            )
        ],
        observation_references=[],
    )


def _plan() -> TeachingPlanDraft:
    """Build a prerequisite-ordered teaching-plan example."""
    return TeachingPlanDraft(
        summary='Build the definition from scalar action before applying it.',
        steps=[
            TeachingPlanStepDraft(
                title='Matrix action and scalar multiplication',
                objective='Interpret Av and explain positive, negative, and zero scalar multiples.',
                prerequisite_positions=[],
            ),
            TeachingPlanStepDraft(
                title='Eigenvector condition',
                objective='Apply Av = lambda v for nonzero v and explain that lambda may be zero.',
                prerequisite_positions=[0],
            ),
        ],
    )


def _question(
    prompt: str, *, answer: str | None = None
) -> TeachingQuestionDraft:
    """Build a fresh free-response question draft."""
    return TeachingQuestionDraft(
        kind=TeachingQuestionKind.CHECK,
        prompt=prompt,
        response=FreeResponseSpecification(reference_answer=answer),
    )


def _question_context(
    question: TeachingQuestionDraft, step: int
) -> TeachingQuestionContext:
    """Build the learner-safe question projection with backend step linkage."""
    return TeachingQuestionContext(
        kind=question.kind,
        prompt=question.prompt,
        response=question.response,
        step_position=step,
    )


def _context(
    *,
    current_assessment: TeachingAssessmentDraft | None = None,
    current_plan: TeachingPlanDraft | None = None,
    current_step_position: int | None = 0,
    pending_question: TeachingQuestionContext | None = None,
    turns: list[TurnMaterial] | None = None,
    blocks: list[SourceBlockMaterial] | None = None,
    discovery: TeachingDiscoveryCatalog | None = None,
    research_enabled: bool = False,
    local_outcomes: list[TeachingLocalOutcome] | None = None,
    citation_candidates: list[TeachingCitationCandidate] | None = None,
) -> TeachingContext:
    """Construct a complete teaching-phase context without storage identities."""
    assessment = current_assessment or _assessment()
    return TeachingContext(
        goal='Understand eigenvectors.',
        phase=TeachingPhase.TEACHING,
        current_assessment=TeachingAssessmentContext(
            summary=assessment.summary,
            strands=assessment.strands,
        ),
        current_plan=current_plan or _plan(),
        current_plan_accepted=True,
        current_step_position=current_step_position,
        pending_question=pending_question,
        material=MaterialContext(
            turns=turns if turns is not None else [],
            blocks=blocks if blocks is not None else [],
        ),
        images=[],
        observations=[],
        tools_remaining=8,
        history_truncated=False,
        research_enabled=research_enabled,
        discovery=discovery
        if discovery is not None
        else TeachingDiscoveryCatalog(),
        local_outcomes=local_outcomes if local_outcomes is not None else [],
        citation_candidates=(
            citation_candidates if citation_candidates is not None else []
        ),
    )


def teacher_examples() -> list[dspy.Example]:
    """Return examples of teaching checks, repair, and consolidation."""
    scalar_check = _question(
        'For nonzero v and scalar c < 0, how does cv compare with v?',
        answer='Its length is scaled by |c| and its orientation is opposite to v.',
    )
    eigenvector_check = _question(
        'For nonzero v with Av = 0, is v an eigenvector, and what is its eigenvalue?',
        answer='Yes. Av = 0v, so v is an eigenvector with eigenvalue zero.',
    )
    local_request = LocalEvidenceRequest(
        question=(
            'How does the diagram show the effect of a negative scalar on a '
            'vector?'
        ),
        context=(
            'The learner is working on scalar geometry and needs the diagram '
            'visually inspected.'
        ),
        require_visual_inspection=True,
    )
    procedure_discovery = TeachingDiscoveryCatalog(
        procedures=[
            TeachingProcedureDiscovery(
                position=0,
                source_key='linear algebra notes',
                preview='A method for testing the eigenvector equation.',
            )
        ]
    )
    inspected_block = SourceBlockMaterial(
        position=0,
        source_key='linear algebra notes',
        page_indexes=[12],
        block_type='paragraph',
        material=MultimodalContent(
            text='A negative scalar reverses orientation and scales length by its absolute value.'
        ),
        asset_positions=[],
        inspected=True,
    )
    turns = [
        TurnMaterial(
            position=0,
            role=TeachingRole.TEACHER,
            material=MultimodalContent(text=scalar_check.prompt),
            question=_question_context(scalar_check, 0),
            responds_to_pending_question=False,
            presents_current_plan=False,
            accepts_current_plan=False,
        ),
        TurnMaterial(
            position=1,
            role=TeachingRole.LEARNER,
            material=MultimodalContent(
                text='Scalar multiplication rotates the vector away from its original line.'
            ),
            question=None,
            responds_to_pending_question=True,
            presents_current_plan=False,
            accepts_current_plan=False,
        ),
    ]
    return [
        dspy.Example(
            context=_context(),
            content_images=[],
            action=AcquireEvidenceAction(request=local_request),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=SearchProceduresAction(
                kind='search_procedures',
                query='Find the method for checking whether a vector is an eigenvector.',
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(discovery=procedure_discovery),
            content_images=[],
            action=ReadProceduresAction(
                kind='read_procedures', procedure_positions=[0]
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                research_enabled=True,
                local_outcomes=[
                    TeachingLocalOutcome(
                        position=0,
                        question=local_request.question,
                        block_positions=[],
                        proposition_positions=[],
                        expanded=False,
                        research=None,
                        require_visual_inspection=True,
                    )
                ],
            ),
            content_images=[],
            action=AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=0),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                pending_question=_question_context(scalar_check, 0),
                turns=turns,
                citation_candidates=[
                    TeachingCitationCandidate(
                        position=0,
                        kind='block',
                        evidence_position=0,
                    )
                ],
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'A scalar multiple remains collinear with v; a '
                            'negative scalar reverses orientation rather than '
                            'rotating it away from its line.'
                        ),
                        cited_evidence_positions=[0],
                    ),
                    TeachingResponseSegment(
                        content=(
                            'For nonzero v and c < 0, how does cv compare '
                            'with v?'
                        ),
                    ),
                ],
                content_kind='instruction',
                operation=RepairCheck(
                    kind='repair_check',
                    question=scalar_check,
                    assessment=_assessment(),
                ),
                learner_observations=[
                    LearnerObservationDraft(
                        kind=LearnerObservationKind.MISCONCEPTION,
                        topic='Scalar multiplication',
                        description='Believes scalar multiplication rotates a vector away from its original line.',
                        evidence_turn_positions=[1],
                    )
                ],
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                current_step_position=0,
                pending_question=_question_context(scalar_check, 0),
                turns=[
                    TurnMaterial(
                        position=0,
                        role=TeachingRole.TEACHER,
                        material=MultimodalContent(text=scalar_check.prompt),
                        question=_question_context(scalar_check, 0),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TurnMaterial(
                        position=1,
                        role=TeachingRole.LEARNER,
                        material=MultimodalContent(
                            text='Its length is scaled by |c|; it points opposite to v and remains collinear with v.'
                        ),
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ],
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'Correct: a negative scalar reverses orientation '
                            'while preserving collinearity and scaling length '
                            'by its absolute value. Now we can connect scalar '
                            'action to the eigenvector definition. For nonzero '
                            'v with Av = 0, is v an eigenvector, and what is '
                            'its eigenvalue?'
                        ),
                    )
                ],
                content_kind='instruction',
                operation=TeachStep(
                    kind='teach_step',
                    step_position=1,
                    question=eigenvector_check,
                ),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                current_step_position=1,
                pending_question=_question_context(eigenvector_check, 1),
                turns=[
                    TurnMaterial(
                        position=0,
                        role=TeachingRole.TEACHER,
                        material=MultimodalContent(
                            text=eigenvector_check.prompt
                        ),
                        question=_question_context(eigenvector_check, 1),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TurnMaterial(
                        position=1,
                        role=TeachingRole.LEARNER,
                        material=MultimodalContent(
                            text='Yes. Av = 0v, so v is an eigenvector with eigenvalue zero.'
                        ),
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ],
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'Exactly. The vector v must be nonzero; Av = 0 '
                            'is Av = 0v, so the eigenvalue is zero.'
                        ),
                    )
                ],
                content_kind='instruction',
                operation=Consolidate(
                    kind='consolidate', assessment=_assessment()
                ),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                pending_question=_question_context(scalar_check, 0),
                turns=[
                    TurnMaterial(
                        position=0,
                        role=TeachingRole.TEACHER,
                        material=MultimodalContent(text=scalar_check.prompt),
                        question=_question_context(scalar_check, 0),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TurnMaterial(
                        position=1,
                        role=TeachingRole.LEARNER,
                        material=MultimodalContent(
                            text='What does collinear mean here? Please give me a hint.'
                        ),
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ],
                blocks=[inspected_block],
                citation_candidates=[
                    TeachingCitationCandidate(
                        position=0,
                        kind='block',
                        evidence_position=0,
                    )
                ],
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'Try separating the effect on length from the '
                            'effect on direction, and use the sign of c to '
                            'guide your reasoning.'
                        ),
                        cited_evidence_positions=[0],
                    )
                ],
                content_kind='instruction',
                operation=Hint(kind='hint'),
            ),
        ).with_inputs('context', 'content_images'),
    ]


__all__ = ['teacher_examples']
