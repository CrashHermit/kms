"""Explicit runtime demonstrations for diagnostic probing."""

from datetime import UTC, datetime

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
    ReadLearnerObservationsAction,
    RespondAction,
    SearchLearnerObservationsAction,
    TeachingContext,
    TeachingDiscoveryCatalog,
    TeachingLearnerObservation,
    TeachingLearnerObservationDiscovery,
    TeachingQuestionContext,
    TeachingToolObservation,
)
from kms.core.model.teaching.observation import LearnerObservationKind
from kms.core.model.teaching.operation import (
    CompleteProbing,
    ContinueProbing,
    TeachingAssessmentDraft,
    TeachingQuestionDraft,
    TeachingResponseSegment,
)
from kms.core.model.teaching.probe import ProbeStrand
from kms.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms.core.model.teaching.session import TeachingPhase
from kms.core.model.teaching.turn import TeachingRole


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
    turns: list[TurnMaterial] | None = None,
    observations: list[TeachingToolObservation] | None = None,
    learner_history: list[TeachingLearnerObservation] | None = None,
    discovery: TeachingDiscoveryCatalog | None = None,
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
        material=MaterialContext(turns=turns if turns is not None else []),
        images=[],
        observations=observations if observations is not None else [],
        learner_history=learner_history if learner_history is not None else [],
        discovery=discovery
        if discovery is not None
        else TeachingDiscoveryCatalog(),
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
    )
    observation = TeachingLearnerObservation(
        position=0,
        registry_position=2,
        kind=LearnerObservationKind.MISCONCEPTION,
        topic='Scalar multiplication',
        description='The learner is unsure how a negative scalar changes vector orientation.',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        evidence=[],
    )
    learner_discovery = TeachingDiscoveryCatalog(
        learner_observations=[
            TeachingLearnerObservationDiscovery(
                position=2,
                kind=LearnerObservationKind.MISCONCEPTION,
                topic='Scalar multiplication',
                preview='The learner is unsure how a negative scalar changes vector orientation.',
            )
        ]
    )
    return [
        dspy.Example(
            context=_context(),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'Before we choose a route, what do you already '
                            'know about how a matrix transforms a vector?'
                        )
                    )
                ],
                content_kind='coordination',
                operation=ContinueProbing(
                    kind='continue_probing', question=_question(prompt)
                ),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=AcquireEvidenceAction(
                request=local_request,
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(),
            content_images=[],
            action=SearchLearnerObservationsAction(
                kind='search_learner_observations',
                query='What is still unclear about scalar multiplication?',
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                learner_history=[observation],
                discovery=learner_discovery,
            ),
            content_images=[],
            action=ReadLearnerObservationsAction(
                kind='read_learner_observations',
                observation_positions=[2],
            ),
        ).with_inputs('context', 'content_images'),
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
            content_images=[],
            action=AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=0),
            ),
        ).with_inputs('context', 'content_images'),
        dspy.Example(
            context=_context(
                turns=[
                    TurnMaterial(
                        position=0,
                        role=TeachingRole.TEACHER,
                        material=MultimodalContent(text=prompt),
                        question=_question_context(prompt),
                        responds_to_pending_question=False,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                    TurnMaterial(
                        position=1,
                        role=TeachingRole.LEARNER,
                        material=MultimodalContent(
                            text='A matrix sends each vector to another vector.'
                        ),
                        question=None,
                        responds_to_pending_question=True,
                        presents_current_plan=False,
                        accepts_current_plan=False,
                    ),
                ]
            ),
            content_images=[],
            action=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content=(
                            'You recognize that a matrix sends vectors to '
                            'vectors. We can now plan a route that connects '
                            'scalar multiples to eigenvectors.'
                        )
                    )
                ],
                content_kind='coordination',
                operation=CompleteProbing(
                    kind='complete_probing', assessment=_assessment()
                ),
            ),
        ).with_inputs('context', 'content_images'),
    ]


__all__ = ['prober_examples']
