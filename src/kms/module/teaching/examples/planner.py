"""Explicit runtime demonstrations for teaching-plan decisions."""

import dspy

from kms.core.model.content import MultimodalContent
from kms.core.model.material import MaterialContext, TurnMaterial
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
    LocalEvidenceRequest,
    TeachingLocalOutcome,
)
from kms.core.model.teaching.inference import (
    ReadArgumentsAction,
    ReadStatementsAction,
    RespondAction,
    SearchArgumentsAction,
    SearchStatementsAction,
    TeachingArgumentDiscovery,
    TeachingContext,
    TeachingDiscoveryCatalog,
    TeachingStatementDiscovery,
)
from kms.core.model.teaching.operation import (
    AcceptPlan,
    ProposePlan,
    TeachingAssessmentContext,
    TeachingAssessmentDraft,
    TeachingPlanDraft,
    TeachingPlanStepDraft,
    TeachingQuestionDraft,
    TeachingResponseSegment,
)
from kms.core.model.teaching.probe import ProbeStrand
from kms.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionContext,
    TeachingQuestionKind,
)
from kms.core.model.teaching.session import TeachingPhase
from kms.core.model.teaching.turn import TeachingRole


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
    turns: list[TurnMaterial] | None = None,
    discovery: TeachingDiscoveryCatalog | None = None,
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
        material=MaterialContext(turns=turns if turns is not None else []),
        images=[],
        observations=[],
        tools_remaining=8,
        history_truncated=False,
        research_enabled=research_enabled,
        discovery=discovery
        if discovery is not None
        else TeachingDiscoveryCatalog(),
        local_outcomes=local_outcomes if local_outcomes is not None else [],
    )


def planner_examples() -> list[dspy.Example]:
    """Return fresh plan-proposal, acceptance, and local-first examples."""
    local_request = LocalEvidenceRequest(
        question='What prerequisites support learning the eigenvector condition?',
        context='The learner recognizes matrix action but is unsure about scalar geometry.',
    )
    argument_discovery = TeachingDiscoveryCatalog(
        arguments=[
            TeachingArgumentDiscovery(
                position=1,
                preview='A structured argument about how scalar multiplication preserves collinearity.',
            )
        ]
    )
    statement_discovery = TeachingDiscoveryCatalog(
        statements=[
            TeachingStatementDiscovery(
                position=0,
                source_key='linear algebra workbook',
                is_exercise=True,
                preview='Determine whether a given vector is an eigenvector.',
            )
        ]
    )
    approval = _question()
    return [
        dspy.Example(
            context=_context(),
            content_images=[],
            action=AcquireEvidenceAction(request=local_request),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=SearchStatementsAction(
                kind='search_statements',
                query='Find an exercise about testing the eigenvector condition.',
                is_exercise=True,
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(discovery=statement_discovery),
            content_images=[],
            action=ReadStatementsAction(
                kind='read_statements', statement_positions=[0]
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=SearchArgumentsAction(
                kind='search_arguments',
                query='Why does scalar multiplication preserve collinearity?',
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(discovery=argument_discovery),
            content_images=[],
            action=ReadArgumentsAction(
                kind='read_arguments',
                argument_positions=[1],
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
                    )
                ],
            ),
            content_images=[],
            action=AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=0),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'I suggest first understanding how scalar '
                            'multiples change a vector, then applying that '
                            'idea to the eigenvector condition. Would you '
                            'like to follow this two-step route?'
                        )
                    )
                ],
                content_kind='coordination',
                operation=ProposePlan(
                    kind='propose_plan', plan=_plan(), question=approval
                ),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                current_plan=_plan(),
                pending_question=_context_question(approval),
                turns=[
                    TurnMaterial(
                        position=0,
                        role=TeachingRole.TEACHER,
                        material=MultimodalContent(text=approval.prompt),
                        question=_context_question(approval),
                        responds_to_pending_question=False,
                        presents_current_plan=True,
                        accepts_current_plan=False,
                    ),
                    TurnMaterial(
                        position=1,
                        role=TeachingRole.LEARNER,
                        material=MultimodalContent(
                            text='Yes, I accept this plan.'
                        ),
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=True,
                    ),
                ],
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'Great, we will use that route and begin with '
                            'how scalar multiples change a vector.'
                        )
                    )
                ],
                content_kind='coordination',
                operation=AcceptPlan(
                    kind='accept_plan',
                    learner_turn_position=1,
                    step_position=0,
                ),
            ),
        ).with_inputs('context', 'content_images'),
    ]


__all__ = ['planner_examples']
