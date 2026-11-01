"""Explicit runtime demonstrations for teaching-plan decisions."""

import dspy

from kms2.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
    LocalEvidenceRequest,
    TeachingLocalOutcome,
)
from kms2.core.model.teaching.inference import (
    RespondAction,
    TeachingContext,
    TeachingContextTurn,
    TeachingQuestionContext,
)
from kms2.core.model.teaching.operation import (
    AcceptPlan,
    ProposePlan,
    TeachingAssessmentContext,
    TeachingAssessmentDraft,
    TeachingPlanDraft,
    TeachingPlanStepDraft,
    TeachingQuestionDraft,
)
from kms2.core.model.teaching.probe import ProbeStrand
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import TeachingPhase
from kms2.core.model.teaching.turn import TeachingRole


def _assessment() -> TeachingAssessmentDraft:
    """Build an assessment from probing evidence."""
    return TeachingAssessmentDraft(
        summary='Recognizes matrix-vector multiplication; scalar geometry is not established.',
        strands=[
            ProbeStrand(
                topic='Eigenvectors',
                demonstrated_understanding='A matrix maps a vector to another vector.',
                observed_limit='The learner is unsure how a scalar changes a vector.',
                gaps=[
                    'How positive, negative, and zero scalars affect a nonzero vector.'
                ],
                misconceptions=[],
            )
        ],
        observation_references=[],
    )


def _plan() -> TeachingPlanDraft:
    """Build a prerequisite-ordered plan example."""
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


def _question() -> TeachingQuestionDraft:
    """Build a plan-approval question."""
    return TeachingQuestionDraft(
        kind=TeachingQuestionKind.PLAN_APPROVAL,
        prompt='Would you like to follow this two-step route?',
        response=FreeResponseSpecification(reference_answer=None),
    )


def _context_question(
    question: TeachingQuestionDraft,
) -> TeachingQuestionContext:
    """Build the learner-safe projection of an approval question."""
    return TeachingQuestionContext(
        kind=question.kind,
        prompt=question.prompt,
        response=question.response,
        step_position=None,
    )


def _context(
    *,
    current_plan: TeachingPlanDraft | None = None,
    pending_question: TeachingQuestionContext | None = None,
    turns: list[TeachingContextTurn] | None = None,
    research_enabled: bool = False,
    local_outcomes: list[TeachingLocalOutcome] | None = None,
) -> TeachingContext:
    """Build a fresh planning context without storage identities."""
    return TeachingContext(
        goal='Understand eigenvectors.',
        phase=TeachingPhase.PLANNING,
        current_assessment=TeachingAssessmentContext(
            summary=_assessment().summary,
            strands=_assessment().strands,
        ),
        current_plan=current_plan,
        current_plan_accepted=False,
        current_step_position=None,
        pending_question=pending_question,
        turns=turns if turns is not None else [],
        blocks=[],
        images=[],
        observations=[],
        tools_remaining=8,
        history_truncated=False,
        research_enabled=research_enabled,
        local_outcomes=local_outcomes if local_outcomes is not None else [],
    )


def planner_examples() -> list[dspy.Example]:
    """Return fresh plan-proposal, acceptance, and local-first examples."""
    local_request = LocalEvidenceRequest(
        question='What prerequisites support learning the eigenvector condition?',
        context='The learner recognizes matrix action but is unsure about scalar geometry.',
        urls=[],
    )
    approval = _question()
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
            context=_context(),
            evidence_images=[],
            action=RespondAction(
                content='I suggest first understanding how scalar multiples change a vector, then applying that idea to the eigenvector condition. Would you like to follow this two-step route?',
                content_kind='coordination',
                operation=ProposePlan(
                    kind='propose_plan', plan=_plan(), question=approval
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                current_plan=_plan(),
                pending_question=_context_question(approval),
                turns=[
                    TeachingContextTurn(
                        position=0,
                        role=TeachingRole.TEACHER,
                        content=approval.prompt,
                        question=_context_question(approval),
                        responds_to_pending_question=False,
                        presents_current_plan=True,
                        accepts_current_plan=False,
                    ),
                    TeachingContextTurn(
                        position=1,
                        role=TeachingRole.LEARNER,
                        content='Yes, I accept this plan.',
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=True,
                    ),
                ],
            ),
            evidence_images=[],
            action=RespondAction(
                content='Great, we will use that route and begin with how scalar multiples change a vector.',
                content_kind='coordination',
                operation=AcceptPlan(
                    kind='accept_plan',
                    learner_turn_position=1,
                    step_position=0,
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
    ]


__all__ = ['planner_examples']
