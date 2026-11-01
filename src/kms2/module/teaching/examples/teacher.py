"""Explicit runtime demonstrations for goal-driven teaching."""

import dspy

from kms2.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
    LocalEvidenceRequest,
    TeachingLocalOutcome,
)
from kms2.core.model.teaching.inference import (
    RespondAction,
    TeachingCitationCandidate,
    TeachingContext,
    TeachingContextBlock,
    TeachingContextTurn,
    TeachingQuestionContext,
)
from kms2.core.model.teaching.observation import (
    LearnerObservationDraft,
    LearnerObservationKind,
)
from kms2.core.model.teaching.operation import (
    Consolidate,
    Hint,
    RepairCheck,
    TeachingAssessmentContext,
    TeachingAssessmentDraft,
    TeachingPlanDraft,
    TeachingPlanStepDraft,
    TeachingQuestionDraft,
    TeachStep,
)
from kms2.core.model.teaching.probe import ProbeStrand
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import TeachingPhase
from kms2.core.model.teaching.turn import TeachingRole


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
    turns: list[TeachingContextTurn] | None = None,
    blocks: list[TeachingContextBlock] | None = None,
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
        turns=turns if turns is not None else [],
        blocks=blocks if blocks is not None else [],
        images=[],
        observations=[],
        tools_remaining=8,
        history_truncated=False,
        research_enabled=research_enabled,
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
        question='How does a negative scalar affect vector direction and length?',
        context='The learner is working on scalar geometry as a prerequisite for eigenvectors.',
        urls=[],
    )
    inspected_block = TeachingContextBlock(
        position=0,
        source_key='linear algebra notes',
        page_indexes=[12],
        block_type='paragraph',
        content='A negative scalar reverses orientation and scales length by its absolute value.',
        image_positions=[],
        inspected=True,
    )
    turns = [
        TeachingContextTurn(
            position=0,
            role=TeachingRole.TEACHER,
            content=scalar_check.prompt,
            question=_question_context(scalar_check, 0),
            responds_to_pending_question=False,
            presents_current_plan=False,
            accepts_current_plan=False,
        ),
        TeachingContextTurn(
            position=1,
            role=TeachingRole.LEARNER,
            content='Scalar multiplication rotates the vector away from its original line.',
            question=None,
            responds_to_pending_question=True,
            presents_current_plan=False,
            accepts_current_plan=False,
        ),
    ]
    return [
        dspy.Example(
            context=_context(),
            evidence_images=[],
            action=AcquireEvidenceAction(request=local_request),
        ).with_inputs('context', 'evidence_images'),
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
                    )
                ],
            ),
            evidence_images=[],
            action=AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=0),
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                pending_question=_question_context(scalar_check, 0),
                turns=turns,
            ),
            evidence_images=[],
            action=RespondAction(
                content='A scalar multiple remains collinear with v; a negative scalar reverses orientation rather than rotating it away from its line. For nonzero v and c < 0, how does cv compare with v?',
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
                cited_evidence_positions=[0],
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                current_step_position=0,
                pending_question=_question_context(scalar_check, 0),
                turns=[
                    TeachingContextTurn(
                        position=0,
                        role=TeachingRole.TEACHER,
                        content=scalar_check.prompt,
                        question=_question_context(scalar_check, 0),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TeachingContextTurn(
                        position=1,
                        role=TeachingRole.LEARNER,
                        content='Its length is scaled by |c|; it points opposite to v and remains collinear with v.',
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ],
            ),
            evidence_images=[],
            action=RespondAction(
                content='Correct: a negative scalar reverses orientation while preserving collinearity and scaling length by its absolute value. Now we can connect scalar action to the eigenvector definition. For nonzero v with Av = 0, is v an eigenvector, and what is its eigenvalue?',
                content_kind='instruction',
                operation=TeachStep(
                    kind='teach_step',
                    step_position=1,
                    question=eigenvector_check,
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                current_step_position=1,
                pending_question=_question_context(eigenvector_check, 1),
                turns=[
                    TeachingContextTurn(
                        position=0,
                        role=TeachingRole.TEACHER,
                        content=eigenvector_check.prompt,
                        question=_question_context(eigenvector_check, 1),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TeachingContextTurn(
                        position=1,
                        role=TeachingRole.LEARNER,
                        content='Yes. Av = 0v, so v is an eigenvector with eigenvalue zero.',
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ],
            ),
            evidence_images=[],
            action=RespondAction(
                content='Exactly. The vector v must be nonzero; Av = 0 is Av = 0v, so the eigenvalue is zero.',
                content_kind='instruction',
                operation=Consolidate(
                    kind='consolidate', assessment=_assessment()
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                pending_question=_question_context(scalar_check, 0),
                turns=[
                    TeachingContextTurn(
                        position=0,
                        role=TeachingRole.TEACHER,
                        content=scalar_check.prompt,
                        question=_question_context(scalar_check, 0),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TeachingContextTurn(
                        position=1,
                        role=TeachingRole.LEARNER,
                        content='What does collinear mean here? Please give me a hint.',
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
            action=RespondAction(
                content='Try separating the effect on length from the effect on direction, and use the sign of c to guide your reasoning.',
                content_kind='instruction',
                operation=Hint(kind='hint'),
                cited_evidence_positions=[0],
            ),
        ).with_inputs('context', 'evidence_images'),
    ]


__all__ = ['teacher_examples']
