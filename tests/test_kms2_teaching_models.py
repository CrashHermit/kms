from datetime import UTC, datetime

from pydantic import TypeAdapter

from kms2.core.model.teaching.evidence import TeachingEvidenceReference
from kms2.core.model.teaching.inference import (
    RespondAction,
    TeachingAction,
    TeachingLearnerObservation,
    TeachingLearnerQuestion,
)
from kms2.core.model.teaching.observation import (
    ExistingLearnerObservationReference,
    LearnerObservationKind,
)
from kms2.core.model.teaching.operation import CompleteProbing
from kms2.core.model.teaching.plan import TeachingPlanAcceptance
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    FreeResponseView,
    MultipleChoiceSpecification,
    TeachingOption,
    TeachingQuestion,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnRecord,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.service.teaching import TeachingService


def test_learner_projection_keeps_question_context_without_answer_metadata():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    multiple_choice = TeachingQuestion(
        uuid='question-history',
        kind=TeachingQuestionKind.CHECK,
        prompt='Which slope is positive?',
        response=MultipleChoiceSpecification(
            options=[
                TeachingOption(key='a', content='Positive'),
                TeachingOption(key='b', content='Negative'),
            ],
            correct_option_key='a',
        ),
        created_at=now,
    )
    free_response = TeachingQuestion(
        uuid='question-pending',
        kind=TeachingQuestionKind.PROBE,
        prompt='Explain instantaneous rate.',
        response=FreeResponseSpecification(
            reference_answer='The derivative at a point.'
        ),
        created_at=now,
    )
    snapshot = TeachingSessionSnapshot(
        session=TeachingSession(
            uuid='session-1',
            goal='Understand derivatives',
            status=TeachingStatus.AWAITING_LEARNER,
            phase=TeachingPhase.PROBING,
            revision=2,
            created_at=now,
            updated_at=now,
        ),
        turns=[
            TeachingTurnRecord(
                turn=TeachingTurn(
                    uuid='teacher-1',
                    role=TeachingRole.TEACHER,
                    content='Choose the slope.',
                    created_at=now,
                ),
                question=multiple_choice,
                question_step_uuid=None,
                responds_to_question_uuid=None,
                presented_plan_uuid=None,
                accepted_plan_uuid=None,
                citations=[
                    TeachingEvidenceReference(kind='block', uuid='block-1')
                ],
            ),
            TeachingTurnRecord(
                turn=TeachingTurn(
                    uuid='teacher-2',
                    role=TeachingRole.TEACHER,
                    content='Now explain the derivative.',
                    created_at=now,
                ),
                question=free_response,
                question_step_uuid=None,
                responds_to_question_uuid=None,
                presented_plan_uuid=None,
                accepted_plan_uuid='plan-1',
            ),
        ],
        current_assessment=None,
        current_plan=None,
        current_plan_acceptance=TeachingPlanAcceptance(
            plan_uuid='plan-1',
            learner_turn_uuid='learner-1',
        ),
        current_step_uuid=None,
        pending_question_uuid=free_response.uuid,
    )

    public = TeachingService._view(snapshot)
    payload = public.model_dump(mode='json')

    assert payload['turns'][0]['question']['response']['options'][0] == {
        'key': 'a',
        'content': 'Positive',
    }
    assert payload['turns'][1]['question']['response'] == {
        'kind': 'free_response'
    }
    assert payload['turns'][1]['accepted_plan_uuid'] == 'plan-1'
    assert payload['pending_question_uuid'] == 'question-pending'
    assert payload['turns'][0]['citations'] == [
        {'kind': 'block', 'uuid': 'block-1'}
    ]
    assert 'reference_answer' not in str(payload)
    assert 'correct_option_key' not in str(payload)
    assert 'The derivative at a point.' not in str(payload)


def test_teacher_action_union_discriminates_operation_and_evidence_positions():
    action = TypeAdapter(TeachingAction).validate_python(
        {
            'kind': 'respond',
            'content': 'A vector is an eigenvector when the matrix scales it.',
            'content_kind': 'instruction',
            'operation': {
                'kind': 'complete_probing',
                'assessment': {
                    'summary': 'Understands the definition.',
                    'strands': [],
                    'observation_references': [
                        {'kind': 'existing', 'position': 0}
                    ],
                },
            },
            'learner_observations': [],
            'cited_evidence_positions': [0],
        }
    )

    assert isinstance(action, RespondAction)
    assert isinstance(action.operation, CompleteProbing)
    assert action.content_kind == 'instruction'
    assert action.cited_evidence_positions == [0]
    assert action.operation.assessment.observation_references == [
        ExistingLearnerObservationReference(kind='existing', position=0)
    ]

    local_action = TypeAdapter(TeachingAction).validate_python(
        {
            'kind': 'acquire_evidence',
            'request': {
                'kind': 'external',
                'local_outcome_position': 0,
            },
        }
    )
    assert local_action.kind == 'acquire_evidence'


def test_learner_history_packet_has_no_storage_or_grading_metadata():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    observation = TeachingLearnerObservation(
        position=0,
        kind=LearnerObservationKind.PARTIAL_UNDERSTANDING,
        topic='slope',
        description='Recognizes rise over run but omits sign.',
        created_at=now,
        evidence=[
            {
                'content': 'The slope is rise over run.',
                'created_at': now,
                'question': TeachingLearnerQuestion(
                    kind=TeachingQuestionKind.CHECK,
                    prompt='How do you find slope?',
                    response=FreeResponseView(),
                ),
            }
        ],
    )

    payload = observation.model_dump(mode='json')
    assert payload['position'] == 0
    assert payload['evidence'][0]['question']['response'] == {
        'kind': 'free_response'
    }
    assert 'uuid' not in str(payload)
    assert 'embedding' not in str(payload)
    assert 'reference_answer' not in str(payload)
    assert 'correct_option_key' not in str(payload)
