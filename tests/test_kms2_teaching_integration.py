import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError

from kms2 import application
from kms2.config.services import DatabaseSettings
from kms2.config.settings import Settings
from kms2.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
    TeachingPlanDependency,
    TeachingPlanStep,
)
from kms2.core.model.teaching.probe import ProbeAssessment, ProbeStrand
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    MultipleChoiceSpecification,
    TeachingOption,
    TeachingQuestion,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingTurnCommit,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.teaching.queries.session import LOAD_TEACHING_SESSION
from kms2.database.teaching.queries.turn import RECORD_TEACHING_TURN
from kms2.database.teaching.repository import TeachingRepository
from kms2.service.teaching import TeachingService

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


pytestmark = pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database=DatabaseSettings(
            uri=os.environ['KMS2_DATABASE__URI'],
            username=os.environ['KMS2_DATABASE__USERNAME'],
            password=os.environ['KMS2_DATABASE__PASSWORD'],
            database=os.environ['KMS2_DATABASE__DATABASE'],
        ),
    )


def _plan(uuid: str, step_prefix: str) -> TeachingPlan:
    first = TeachingPlanStep(
        uuid=f'{step_prefix}-a',
        title='Foundation',
        objective='Explain the foundation',
    )
    second = TeachingPlanStep(
        uuid=f'{step_prefix}-b',
        title='Consequence',
        objective='Use the foundation',
    )
    independent = TeachingPlanStep(
        uuid=f'{step_prefix}-c',
        title='Bridge',
        objective='Connect the ideas',
    )
    return TeachingPlan(
        uuid=uuid,
        summary=f'Plan {uuid}',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        steps=[second, independent, first],
        dependencies=[
            TeachingPlanDependency(
                step_uuid=second.uuid,
                prerequisite_uuid=first.uuid,
            )
        ],
    )


def _commit(
    *,
    plan: TeachingPlan | None,
    phase: TeachingPhase,
    current_step_uuid: str | None,
    pending_question_uuid: str | None,
    current_plan_uuid: str | None = None,
    assessment: ProbeAssessment | None = None,
    current_assessment_uuid: str | None = None,
    question: TeachingQuestion | None = None,
    question_step_uuid: str | None = None,
    acceptance: TeachingPlanAcceptance | None = None,
) -> TeachingTurnCommit:
    return TeachingTurnCommit(
        phase=phase,
        current_assessment_uuid=(
            assessment.uuid if assessment else current_assessment_uuid
        ),
        current_plan_uuid=current_plan_uuid or (plan.uuid if plan else None),
        current_step_uuid=current_step_uuid,
        pending_question_uuid=pending_question_uuid,
        new_assessment=assessment,
        new_plan=plan,
        new_question=question,
        new_question_step_uuid=question_step_uuid,
        plan_acceptance=acceptance,
    )


async def _owned_user_fixture(
    settings: Settings,
    suffix: str,
):
    return await application.create_user(
        settings, f'Teaching integration {suffix}'
    )


async def _delete_fixture(
    database: DatabaseClient,
    user_uuids: list[str],
    session_uuids: list[str],
    extra_uuids: list[str],
) -> None:
    async with database.session() as session:
        result = await session.run(
            """
            MATCH (session:TeachingSession)
            WHERE session.uuid IN $session_uuids
            OPTIONAL MATCH (session)-[:HAS_TURN]->(turn:TeachingTurn)
            OPTIONAL MATCH (turn)-[:ASKS]->(question:TeachingQuestion)
            OPTIONAL MATCH (session)-[:HAS_PLAN]->(plan:TeachingPlan)
            OPTIONAL MATCH (plan)-[:HAS_STEP]->(step:TeachingPlanStep)
            OPTIONAL MATCH (session)-[:HAS_ASSESSMENT]->
                (assessment:ProbeAssessment)
            WITH collect(DISTINCT session)
                 + collect(DISTINCT turn)
                 + collect(DISTINCT question)
                 + collect(DISTINCT plan)
                 + collect(DISTINCT step)
                 + collect(DISTINCT assessment) AS nodes
            UNWIND nodes AS node
            WITH node WHERE node IS NOT NULL
            DETACH DELETE node
            """,
            session_uuids=session_uuids,
        )
        await result.consume()
        result = await session.run(
            """
            MATCH (node)
            WHERE node.uuid IN $extra_uuids
            DETACH DELETE node
            """,
            extra_uuids=extra_uuids,
        )
        await result.consume()
        result = await session.run(
            """
            MATCH (user:User)
            WHERE user.uuid IN $user_uuids
            OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
            OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                (settings:UserSchedulerSettings)
            WITH collect(user) + collect(deck) + collect(settings) AS nodes
            UNWIND nodes AS node
            WITH node WHERE node IS NOT NULL
            DETACH DELETE node
            """,
            user_uuids=user_uuids,
        )
        await result.consume()


class _FailingTransaction:
    def __init__(self, transaction):
        self._transaction = transaction
        self._armed = False

    async def __aenter__(self):
        await self._transaction.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self._transaction.__aexit__(exc_type, exc_value, traceback)

    async def run(self, query, **parameters):
        if self._armed and query == LOAD_TEACHING_SESSION:
            raise RuntimeError('teaching persistence interrupted')
        result = await self._transaction.run(query, **parameters)
        if query == RECORD_TEACHING_TURN:
            self._armed = True
        return result


class _FailingSession:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self._session.__aexit__(exc_type, exc_value, traceback)

    async def begin_transaction(self):
        return _FailingTransaction(await self._session.begin_transaction())


def test_teaching_sessions_persist_and_preserve_history():
    async def exercise() -> None:
        settings = _settings()
        suffix = uuid4().hex
        database = DatabaseClient(settings.database)
        repository = TeachingRepository(database.session)
        service = TeachingService(repository)
        user_a = None
        user_b = None
        teaching_uuids: list[str] = []
        try:
            await schema.ensure_structural_schema(database.session)
            user_a = await _owned_user_fixture(settings, suffix + '-a')
            user_b = await _owned_user_fixture(settings, suffix + '-b')
            sentinel_source_uuid = f'{suffix}-sentinel-source'
            sentinel_card_uuid = f'{suffix}-sentinel-card'
            sentinel_review_uuid = f'{suffix}-sentinel-review'
            sentinel_event_uuid = f'{suffix}-sentinel-event'
            teaching_uuids.extend(
                [
                    sentinel_source_uuid,
                    sentinel_card_uuid,
                    sentinel_review_uuid,
                    sentinel_event_uuid,
                ]
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    CREATE (source:Source {
                        uuid: $source_uuid,
                        key: 'teaching-sentinel.pdf'
                    })
                    CREATE (card:SourceFlashcard {
                        uuid: $card_uuid,
                        question: 'Sentinel question',
                        answer: 'Sentinel answer'
                    })
                    CREATE (review:UserCardReview {
                        uuid: $review_uuid,
                        fsrs_card_json: '{}',
                        due_at: $due_at,
                        review_count: 0
                    })
                    CREATE (event:ReviewEvent {
                        uuid: $event_uuid,
                        review_index: 1,
                        scheduler_json: '{}',
                        fsrs_card_before_json: '{}',
                        fsrs_card_after_json: '{}',
                        fsrs_review_log_json: '{}'
                    })
                    CREATE (user)-[:OWNS_SOURCE]->(source)
                    CREATE (source)-[:HAS_FLASHCARD]->(card)
                    CREATE (user)-[:HAS_CARD_REVIEW]->(review)
                    CREATE (card)-[:HAS_CARD_REVIEW]->(review)
                    CREATE (review)-[:HAS_REVIEW_EVENT]->(event)
                    RETURN source.key AS source_key,
                           card.question AS card_question,
                           review.review_count AS review_count,
                           event.review_index AS review_index
                    """,
                    user_uuid=user_a.uuid,
                    source_uuid=sentinel_source_uuid,
                    card_uuid=sentinel_card_uuid,
                    review_uuid=sentinel_review_uuid,
                    event_uuid=sentinel_event_uuid,
                    due_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
                sentinel_before = await result.single(strict=True)
            assert dict(sentinel_before) == {
                'source_key': 'teaching-sentinel.pdf',
                'card_question': 'Sentinel question',
                'review_count': 0,
                'review_index': 1,
            }
            session_view = await application.start_teaching_session(
                settings,
                user_a.uuid,
                'Understand why a negative derivative implies local decrease',
            )
            session_uuid = session_view.session.uuid
            teaching_uuids.append(session_uuid)
            assert session_view.session.revision == 0
            assert session_view.turns == []
            reloaded = await application.load_teaching_session(
                settings,
                user_a.uuid,
                session_uuid,
            )
            assert reloaded.session.goal == session_view.session.goal
            assert reloaded.session.revision == 0
            assert (
                await application.list_teaching_sessions(settings, user_b.uuid)
                == []
            )
            with pytest.raises(ResultNotSingleError):
                await application.load_teaching_session(
                    settings,
                    user_b.uuid,
                    session_uuid,
                )
            with pytest.raises(ResultNotSingleError):
                await repository.append_response(
                    user_b.uuid,
                    session_uuid,
                    0,
                    TeachingTurn(
                        role=TeachingRole.LEARNER,
                        content='Foreign response',
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                )
            with pytest.raises(ResultNotSingleError):
                await repository.record_turn(
                    user_b.uuid,
                    session_uuid,
                    0,
                    TeachingTurn(
                        role=TeachingRole.TEACHER,
                        content='Foreign teacher turn',
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                    TeachingTurnCommit(
                        phase=TeachingPhase.PROBING,
                        current_assessment_uuid=None,
                        current_plan_uuid=None,
                        current_step_uuid=None,
                        pending_question_uuid=None,
                    ),
                )
            with pytest.raises(ResultNotSingleError):
                await repository.end_session(
                    user_b.uuid,
                    session_uuid,
                    0,
                    datetime(2026, 1, 1, tzinfo=UTC),
                )

            plan = _plan(f'{suffix}-plan-1', f'{suffix}-step-1')
            assessment = ProbeAssessment(
                uuid=f'{suffix}-assessment-1',
                summary='The learner knows average slope.',
                strands=[
                    ProbeStrand(
                        topic='derivative',
                        demonstrated_understanding='average slope',
                        observed_limit='instantaneous rate',
                        gaps=['local rate'],
                        misconceptions=[],
                    )
                ],
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            multiple_choice = TeachingQuestion(
                uuid=f'{suffix}-question-multiple-choice',
                kind=TeachingQuestionKind.PLAN_APPROVAL,
                prompt='Which direction is the derivative?',
                response=MultipleChoiceSpecification(
                    options=[
                        TeachingOption(key='a', content='Increasing'),
                        TeachingOption(key='b', content='Decreasing'),
                    ],
                    correct_option_key='b',
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            teaching_uuids.extend(
                [
                    plan.uuid,
                    assessment.uuid,
                    multiple_choice.uuid,
                    *(step.uuid for step in plan.steps),
                ]
            )
            first_teacher = await service.record_teaching_turn(
                user_a.uuid,
                session_uuid,
                'Let us first probe your understanding.',
                _commit(
                    plan=plan,
                    phase=TeachingPhase.PLANNING,
                    current_step_uuid=f'{suffix}-step-1-a',
                    pending_question_uuid=multiple_choice.uuid,
                    assessment=assessment,
                    question=multiple_choice,
                    question_step_uuid=f'{suffix}-step-1-a',
                ),
                expected_revision=0,
            )
            assert [step.uuid for step in first_teacher.current_plan.steps] == [
                f'{suffix}-step-1-a',
                f'{suffix}-step-1-c',
                f'{suffix}-step-1-b',
            ]
            public_after_teacher = await application.load_teaching_session(
                settings,
                user_a.uuid,
                session_uuid,
            )
            public_payload = public_after_teacher.model_dump(mode='json')
            assert 'correct_option_key' not in str(public_payload)
            assert 'Increasing' in str(public_payload)

            response = await application.submit_teaching_response(
                settings,
                user_a.uuid,
                session_uuid,
                'I understand average slope but not instantaneous rate.',
                expected_revision=1,
            )
            learner_uuid = response.turns[-1].turn.uuid
            teaching_uuids.append(learner_uuid)
            assert response.session.status.value == 'awaiting_teacher'
            assert response.turns[-1].responds_to_question_uuid == (
                multiple_choice.uuid
            )
            assert response.pending_question_uuid == multiple_choice.uuid

            accepted = await service.record_teaching_turn(
                user_a.uuid,
                session_uuid,
                'The plan is accepted; we can teach it.',
                _commit(
                    plan=None,
                    phase=TeachingPhase.TEACHING,
                    current_step_uuid=f'{suffix}-step-1-b',
                    pending_question_uuid=None,
                    current_plan_uuid=plan.uuid,
                    current_assessment_uuid=assessment.uuid,
                    acceptance=TeachingPlanAcceptance(
                        plan_uuid=plan.uuid,
                        learner_turn_uuid=learner_uuid,
                    ),
                ),
                expected_revision=2,
            )
            assert accepted.current_plan_acceptance is not None
            assert (
                accepted.current_plan_acceptance.learner_turn_uuid
                == learner_uuid
            )
            assert accepted.turns[1].accepted_plan_uuid == plan.uuid
            accepted_public = await application.load_teaching_session(
                settings,
                user_a.uuid,
                session_uuid,
            )
            assert accepted_public.current_plan_acceptance == (
                accepted.current_plan_acceptance
            )
            first_plan_uuid = plan.uuid

            second_response = await application.submit_teaching_response(
                settings,
                user_a.uuid,
                session_uuid,
                'I accept the sequence.',
                expected_revision=3,
            )
            assert second_response.turns[-1].responds_to_question_uuid is None
            preserved = await service.record_teaching_turn(
                user_a.uuid,
                session_uuid,
                'Continue with the accepted plan.',
                _commit(
                    plan=None,
                    phase=TeachingPhase.TEACHING,
                    current_step_uuid=f'{suffix}-step-1-b',
                    pending_question_uuid=None,
                    current_plan_uuid=first_plan_uuid,
                    current_assessment_uuid=assessment.uuid,
                ),
                expected_revision=4,
            )
            assert (
                preserved.current_plan_acceptance.plan_uuid == first_plan_uuid
            )

            replacement_plan = _plan(f'{suffix}-plan-2', f'{suffix}-step-2')
            free_response = TeachingQuestion(
                uuid=f'{suffix}-question-free-response',
                kind=TeachingQuestionKind.CLARIFICATION,
                prompt='Explain instantaneous rate.',
                response=FreeResponseSpecification(
                    reference_answer='The derivative at one point.'
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            teaching_uuids.extend(
                [
                    replacement_plan.uuid,
                    free_response.uuid,
                    *(step.uuid for step in replacement_plan.steps),
                ]
            )
            await application.submit_teaching_response(
                settings,
                user_a.uuid,
                session_uuid,
                'I need a different explanation.',
                expected_revision=5,
            )
            replacement = await service.record_teaching_turn(
                user_a.uuid,
                session_uuid,
                'We will revise the plan and clarify the concept.',
                _commit(
                    plan=replacement_plan,
                    phase=TeachingPhase.PLANNING,
                    current_step_uuid=f'{suffix}-step-2-a',
                    pending_question_uuid=free_response.uuid,
                    current_assessment_uuid=assessment.uuid,
                    question=free_response,
                    question_step_uuid=f'{suffix}-step-2-a',
                ),
                expected_revision=6,
            )
            assert replacement.current_plan.uuid == replacement_plan.uuid
            assert replacement.current_plan_acceptance is None
            clarification = await application.submit_teaching_response(
                settings,
                user_a.uuid,
                session_uuid,
                'I need clarification on the derivative.',
                expected_revision=7,
            )
            assert clarification.pending_question_uuid == free_response.uuid
            assert clarification.turns[-1].responds_to_question_uuid == (
                free_response.uuid
            )

            replacement_question = TeachingQuestion(
                uuid=f'{suffix}-question-replacement',
                kind=TeachingQuestionKind.CHECK,
                prompt='State the local-decrease implication.',
                response=MultipleChoiceSpecification(
                    options=[TeachingOption(key='a', content='Decrease')],
                    correct_option_key='a',
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            teaching_uuids.append(replacement_question.uuid)
            await service.record_teaching_turn(
                user_a.uuid,
                session_uuid,
                'Try this more focused check.',
                _commit(
                    plan=None,
                    phase=TeachingPhase.TEACHING,
                    current_step_uuid=f'{suffix}-step-2-a',
                    pending_question_uuid=replacement_question.uuid,
                    current_plan_uuid=replacement_plan.uuid,
                    current_assessment_uuid=assessment.uuid,
                    question=replacement_question,
                    question_step_uuid=f'{suffix}-step-2-a',
                ),
                expected_revision=8,
            )
            final_response = await application.submit_teaching_response(
                settings,
                user_a.uuid,
                session_uuid,
                'A negative derivative means the function decreases locally.',
                expected_revision=9,
            )
            assert final_response.turns[-1].responds_to_question_uuid == (
                replacement_question.uuid
            )
            assert final_response.session.revision == 10
            final_payload = final_response.model_dump(mode='json')
            assert 'correct_option_key' not in str(final_payload)
            assert 'The derivative at one point.' not in str(final_payload)

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (session:TeachingSession {uuid: $session_uuid})
                    MATCH (session)-[:HAS_PLAN]->(plan:TeachingPlan)
                    OPTIONAL MATCH (session)-[:HAS_TURN]->(turn:TeachingTurn)
                    OPTIONAL MATCH (turn)-[:PRESENTS_PLAN]->
                        (presented:TeachingPlan)
                    RETURN collect(DISTINCT plan.uuid) AS plans,
                           count(DISTINCT presented) AS presented_count,
                           count(DISTINCT turn) AS turn_count
                    """,
                    session_uuid=session_uuid,
                )
                topology = await result.single(strict=True)
            assert set(topology['plans']) == {plan.uuid, replacement_plan.uuid}
            assert topology['presented_count'] == 2
            assert topology['turn_count'] == 10

            ended = await application.end_teaching_session(
                settings,
                user_a.uuid,
                session_uuid,
                expected_revision=10,
            )
            assert ended.session.status.value == 'ended'
            assert ended.turns
            assert ended.pending_question_uuid is None

            teacher_waiting_session = await application.start_teaching_session(
                settings,
                user_a.uuid,
                f'End while awaiting teacher {suffix}',
            )
            teacher_end = await application.end_teaching_session(
                settings,
                user_a.uuid,
                teacher_waiting_session.session.uuid,
                expected_revision=0,
            )
            assert teacher_end.session.status.value == 'ended'

            learner_waiting_session = await application.start_teaching_session(
                settings,
                user_a.uuid,
                f'End while awaiting learner {suffix}',
            )
            learner_waiting_service = TeachingService(
                TeachingRepository(database.session)
            )
            await learner_waiting_service.record_teaching_turn(
                user_a.uuid,
                learner_waiting_session.session.uuid,
                'One teacher message.',
                TeachingTurnCommit(
                    phase=TeachingPhase.PROBING,
                    current_assessment_uuid=None,
                    current_plan_uuid=None,
                    current_step_uuid=None,
                    pending_question_uuid=None,
                ),
                expected_revision=0,
            )
            learner_end = await application.end_teaching_session(
                settings,
                user_a.uuid,
                learner_waiting_session.session.uuid,
                expected_revision=1,
            )
            assert learner_end.session.status.value == 'ended'

            constraints = set()
            async with database.session() as session:
                result = await session.run(
                    """
                    SHOW CONSTRAINTS YIELD name
                    WHERE name IN [
                        'teaching_session_uuid',
                        'teaching_turn_uuid',
                        'probe_assessment_uuid',
                        'teaching_plan_uuid',
                        'teaching_plan_step_uuid',
                        'teaching_question_uuid'
                    ]
                    RETURN collect(name) AS names
                    """
                )
                constraints = set((await result.single(strict=True))['names'])
            assert constraints == {
                'teaching_session_uuid',
                'teaching_turn_uuid',
                'probe_assessment_uuid',
                'teaching_plan_uuid',
                'teaching_plan_step_uuid',
                'teaching_question_uuid',
            }
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
                    MATCH (card)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
                    MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
                    RETURN source.key AS source_key,
                           card.question AS card_question,
                           review.review_count AS review_count,
                           event.review_index AS review_index
                    """,
                    source_uuid=sentinel_source_uuid,
                )
                sentinel_after = await result.single(strict=True)
            assert dict(sentinel_after) == dict(sentinel_before)
        finally:
            teaching_uuids.extend(
                [
                    uuid
                    for uuid in (
                        session_uuid if 'session_uuid' in locals() else None,
                        (
                            teacher_waiting_session.session.uuid
                            if 'teacher_waiting_session' in locals()
                            else None
                        ),
                        (
                            learner_waiting_session.session.uuid
                            if 'learner_waiting_session' in locals()
                            else None
                        ),
                    )
                    if uuid is not None
                ]
            )
            await _delete_fixture(
                database,
                [user.uuid for user in (user_a, user_b) if user is not None],
                [
                    uuid
                    for uuid in (
                        session_uuid if 'session_uuid' in locals() else None,
                        (
                            teacher_waiting_session.session.uuid
                            if 'teacher_waiting_session' in locals()
                            else None
                        ),
                        (
                            learner_waiting_session.session.uuid
                            if 'learner_waiting_session' in locals()
                            else None
                        ),
                    )
                    if uuid is not None
                ],
                list(set(teaching_uuids)),
            )
            await database.close()

    asyncio.run(exercise())


def test_teaching_session_revision_conflict_and_rollback():
    async def exercise() -> None:
        settings = _settings()
        suffix = uuid4().hex
        database = DatabaseClient(settings.database)
        user = None
        teaching_uuids: list[str] = []
        try:
            user = await _owned_user_fixture(settings, suffix)
            service = TeachingService(TeachingRepository(database.session))
            started = await application.start_teaching_session(
                settings,
                user.uuid,
                'Concurrency goal',
            )
            session_uuid = started.session.uuid
            teaching_uuids.append(session_uuid)
            commit = TeachingTurnCommit(
                phase=TeachingPhase.PROBING,
                current_assessment_uuid=None,
                current_plan_uuid=None,
                current_step_uuid=None,
                pending_question_uuid=None,
            )
            results = await asyncio.gather(
                service.record_teaching_turn(
                    user.uuid,
                    session_uuid,
                    'Teacher one',
                    commit,
                    expected_revision=0,
                ),
                service.record_teaching_turn(
                    user.uuid,
                    session_uuid,
                    'Teacher two',
                    commit,
                    expected_revision=0,
                ),
                return_exceptions=True,
            )
            successes = [
                result
                for result in results
                if not isinstance(result, Exception)
            ]
            failures = [
                result for result in results if isinstance(result, Exception)
            ]
            assert len(successes) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], ResultNotSingleError)
            current = await service.load_session(user.uuid, session_uuid)
            assert current.session.revision == 1
            assert len(current.turns) == 1
            race_start = await application.start_teaching_session(
                settings,
                user.uuid,
                'End versus teacher commit',
            )
            race_uuid = race_start.session.uuid
            teaching_uuids.append(race_uuid)
            race_results = await asyncio.gather(
                application.end_teaching_session(
                    settings,
                    user.uuid,
                    race_uuid,
                    expected_revision=0,
                ),
                service.record_teaching_turn(
                    user.uuid,
                    race_uuid,
                    'A competing teacher turn.',
                    commit,
                    expected_revision=0,
                ),
                return_exceptions=True,
            )
            race_successes = [
                result
                for result in race_results
                if not isinstance(result, Exception)
            ]
            race_failures = [
                result
                for result in race_results
                if isinstance(result, Exception)
            ]
            assert len(race_successes) == 1
            assert len(race_failures) == 1
            assert isinstance(race_failures[0], ResultNotSingleError)
            race_state = await service.load_session(user.uuid, race_uuid)
            assert race_state.session.status.value in {
                'ended',
                'awaiting_learner',
            }
            if race_state.session.status.value == 'ended':
                assert race_state.turns == []
            else:
                assert len(race_state.turns) == 1

            rollback_start = await application.start_teaching_session(
                settings,
                user.uuid,
                'Rollback goal',
            )
            rollback_uuid = rollback_start.session.uuid
            teaching_uuids.append(rollback_uuid)
            before = await service.load_session(user.uuid, rollback_uuid)
            failing_database = DatabaseClient(settings.database)

            def failing_factory():
                return _FailingSession(failing_database.session())

            failing_repository = TeachingRepository(failing_factory)
            with pytest.raises(
                RuntimeError,
                match='teaching persistence interrupted',
            ):
                await failing_repository.record_turn(
                    user.uuid,
                    rollback_uuid,
                    0,
                    TeachingTurn(
                        role=TeachingRole.TEACHER,
                        content='This must roll back.',
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                    commit,
                )
            after = await service.load_session(user.uuid, rollback_uuid)
            assert after == before
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (session:TeachingSession {uuid: $session_uuid})
                    OPTIONAL MATCH (session)-[:HAS_TURN]->(turn:TeachingTurn)
                    RETURN session.revision AS revision,
                           session._teaching_write_lock AS write_lock,
                           count(turn) AS turn_count
                    """,
                    session_uuid=rollback_uuid,
                )
                row = await result.single(strict=True)
            assert row['revision'] == 0
            assert row['write_lock'] is None
            assert row['turn_count'] == 0
            await failing_database.close()
        finally:
            await _delete_fixture(
                database,
                [user.uuid] if user is not None else [],
                [
                    uuid
                    for uuid in (
                        session_uuid if 'session_uuid' in locals() else None,
                        race_uuid if 'race_uuid' in locals() else None,
                        rollback_uuid if 'rollback_uuid' in locals() else None,
                    )
                    if uuid is not None
                ],
                list(set(teaching_uuids)),
            )
            await database.close()

    asyncio.run(exercise())
