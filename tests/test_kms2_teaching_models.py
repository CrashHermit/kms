from datetime import UTC, datetime

from kms2.core.model.teaching.plan import TeachingPlanAcceptance
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
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
    assert 'correct_option_key' not in str(payload)
    assert 'The derivative at a point.' not in str(payload)
