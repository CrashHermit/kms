"""Explicit runtime demonstrations for diagnostic probing."""

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
    TeachingToolObservation,
)
from kms2.core.model.teaching.operation import (
    CompleteProbing,
    ContinueProbing,
    TeachingAssessmentDraft,
    TeachingQuestionDraft,
)
from kms2.core.model.teaching.probe import ProbeStrand
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import TeachingPhase
from kms2.core.model.teaching.turn import TeachingRole


def _question(prompt: str) -> TeachingQuestionDraft:
    """Build a fresh free-response diagnostic question."""
    return TeachingQuestionDraft(
        kind=TeachingQuestionKind.PROBE,
        prompt=prompt,
        response=FreeResponseSpecification(reference_answer=None),
    )


def _question_context(prompt: str) -> TeachingQuestionContext:
    """Build its learner-safe context projection."""
    question = _question(prompt)
    return TeachingQuestionContext(
        kind=question.kind,
        prompt=question.prompt,
        response=question.response,
        step_position=None,
    )


def _assessment() -> TeachingAssessmentDraft:
    """Build an assessment based on observed diagnostic evidence."""
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


def _context(
    *,
    pending_question: TeachingQuestionContext | None = None,
    turns: list[TeachingContextTurn] | None = None,
    observations: list[TeachingToolObservation] | None = None,
    research_enabled: bool = False,
    local_outcomes: list[TeachingLocalOutcome] | None = None,
) -> TeachingContext:
    """Build a fresh probing context without storage identities."""
    return TeachingContext(
        goal='Understand eigenvectors.',
        phase=TeachingPhase.PROBING,
        current_assessment=None,
        current_plan=None,
        current_plan_accepted=False,
        current_step_position=None,
        pending_question=pending_question,
        turns=turns if turns is not None else [],
        blocks=[],
        images=[],
        observations=observations if observations is not None else [],
        tools_remaining=8,
        history_truncated=False,
        research_enabled=research_enabled,
        local_outcomes=local_outcomes if local_outcomes is not None else [],
    )


def prober_examples() -> list[dspy.Example]:
    """Return fresh diagnostic and local-first acquisition examples."""
    prompt = 'What do you already know about how a matrix transforms a vector?'
    local_request = LocalEvidenceRequest(
        question='What local explanation supports diagnosing eigenvector prerequisites?',
        context='The learner is beginning to study eigenvectors.',
        urls=[],
    )
    return [
        dspy.Example(
            context=_context(),
            evidence_images=[],
            action=RespondAction(
                content='Before we choose a route, what do you already know about how a matrix transforms a vector?',
                content_kind='coordination',
                operation=ContinueProbing(
                    kind='continue_probing', question=_question(prompt)
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(),
            evidence_images=[],
            action=AcquireEvidenceAction(
                request=local_request,
            ),
        ).with_inputs('context', 'evidence_images'),
        dspy.Example(
            context=_context(
                research_enabled=True,
                local_outcomes=[
                    TeachingLocalOutcome(
                        position=0,
                        question=local_request.question,
                        block_positions=[0],
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
                turns=[
                    TeachingContextTurn(
                        position=0,
                        role=TeachingRole.TEACHER,
                        content=prompt,
                        question=_question_context(prompt),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TeachingContextTurn(
                        position=1,
                        role=TeachingRole.LEARNER,
                        content='A matrix sends each vector to another vector.',
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ]
            ),
            evidence_images=[],
            action=RespondAction(
                content='You recognize that a matrix sends vectors to vectors. We can now plan a route that connects scalar multiples to eigenvectors.',
                content_kind='coordination',
                operation=CompleteProbing(
                    kind='complete_probing', assessment=_assessment()
                ),
            ),
        ).with_inputs('context', 'evidence_images'),
    ]


__all__ = ['prober_examples']
