"""Transactional persistence for user-scoped teaching sessions."""

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from graphlib import TopologicalSorter
from typing import Any

from neo4j import AsyncSession, AsyncTransaction, Record
from neo4j.time import DateTime

from kms2.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
    TeachingPlanDependency,
    TeachingPlanStep,
)
from kms2.core.model.teaching.probe import ProbeAssessment, ProbeStrand
from kms2.core.model.teaching.question import (
    TeachingQuestion,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import (
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnCommit,
    TeachingTurnRecord,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.database.teaching.queries.session import (
    CREATE_TEACHING_SESSION,
    END_TEACHING_SESSION,
    LOAD_TEACHING_SESSION,
    LOCK_TEACHING_SESSION,
    READ_TEACHING_SESSION_REVISION,
    READ_TEACHING_SESSIONS,
)
from kms2.database.teaching.queries.turn import (
    APPEND_TEACHING_RESPONSE,
    RECORD_TEACHING_TURN,
)


class TeachingRepository:
    """Persist immutable teaching history and locked session state."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_session(
        self,
        user_uuid: str,
        session: TeachingSession,
    ) -> TeachingSessionSnapshot:
        """Create an empty session owned by one existing user."""
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._single(
                    transaction,
                    CREATE_TEACHING_SESSION,
                    user_uuid=user_uuid,
                    session_uuid=session.uuid,
                    goal=session.goal,
                    status=session.status.value,
                    phase=session.phase.value,
                    revision=session.revision,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    ended_at=session.ended_at,
                )
                return await self._load(transaction, user_uuid, session.uuid)

    async def list_sessions(self, user_uuid: str) -> list[TeachingSession]:
        """List one user's sessions in reverse update order."""
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_TEACHING_SESSIONS,
                user_uuid=user_uuid,
            )
            rows = await result.data()
        return [_parse_session(row) for row in rows]

    async def load_session(
        self,
        user_uuid: str,
        session_uuid: str,
    ) -> TeachingSessionSnapshot:
        """Load one complete private teaching snapshot."""
        async with self._session_factory() as database_session:
            result = await database_session.run(
                LOAD_TEACHING_SESSION,
                user_uuid=user_uuid,
                session_uuid=session_uuid,
            )
            row = await result.single(strict=True)
        return _parse_snapshot(row)

    async def append_response(
        self,
        user_uuid: str,
        session_uuid: str,
        expected_revision: int,
        turn: TeachingTurn,
    ) -> TeachingSessionSnapshot:
        """Append one learner response through an atomic revision transition."""
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_and_check(
                    transaction,
                    user_uuid,
                    session_uuid,
                    expected_revision,
                    [TeachingStatus.AWAITING_LEARNER.value],
                )
                await self._single(
                    transaction,
                    APPEND_TEACHING_RESPONSE,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    turn_uuid=turn.uuid,
                    content=turn.content,
                    created_at=turn.created_at,
                    new_revision=expected_revision + 1,
                )
                return await self._load(transaction, user_uuid, session_uuid)

    async def record_turn(
        self,
        user_uuid: str,
        session_uuid: str,
        expected_revision: int,
        turn: TeachingTurn,
        commit: TeachingTurnCommit,
    ) -> TeachingSessionSnapshot:
        """Persist one teacher turn and its complete intended session state."""
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_and_check(
                    transaction,
                    user_uuid,
                    session_uuid,
                    expected_revision,
                    [TeachingStatus.AWAITING_TEACHER.value],
                )
                await self._single(
                    transaction,
                    RECORD_TEACHING_TURN,
                    **_record_parameters(
                        user_uuid=user_uuid,
                        session_uuid=session_uuid,
                        expected_revision=expected_revision,
                        turn=turn,
                        commit=commit,
                    ),
                )
                return await self._load(transaction, user_uuid, session_uuid)

    async def end_session(
        self,
        user_uuid: str,
        session_uuid: str,
        expected_revision: int,
        ended_at: datetime,
    ) -> TeachingSessionSnapshot:
        """End one active session without deleting its immutable history."""
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_and_check(
                    transaction,
                    user_uuid,
                    session_uuid,
                    expected_revision,
                    [
                        TeachingStatus.AWAITING_LEARNER.value,
                        TeachingStatus.AWAITING_TEACHER.value,
                    ],
                )
                await self._single(
                    transaction,
                    END_TEACHING_SESSION,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    ended_at=ended_at,
                    new_revision=expected_revision + 1,
                )
                return await self._load(transaction, user_uuid, session_uuid)

    @staticmethod
    async def _single(
        transaction: AsyncTransaction,
        query: str,
        **parameters: object,
    ) -> Record:
        result = await transaction.run(query, **parameters)
        return await result.single(strict=True)

    async def _lock_and_check(
        self,
        transaction: AsyncTransaction,
        user_uuid: str,
        session_uuid: str,
        expected_revision: int,
        allowed_statuses: list[str],
    ) -> None:
        await self._single(
            transaction,
            LOCK_TEACHING_SESSION,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
        )
        await self._single(
            transaction,
            READ_TEACHING_SESSION_REVISION,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
            expected_revision=expected_revision,
            allowed_statuses=allowed_statuses,
        )

    @staticmethod
    async def _load(
        transaction: AsyncTransaction,
        user_uuid: str,
        session_uuid: str,
    ) -> TeachingSessionSnapshot:
        result = await transaction.run(
            LOAD_TEACHING_SESSION,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
        )
        return _parse_snapshot(await result.single(strict=True))


def _record_parameters(
    *,
    user_uuid: str,
    session_uuid: str,
    expected_revision: int,
    turn: TeachingTurn,
    commit: TeachingTurnCommit,
) -> dict[str, object]:
    """Convert typed teacher state into one Cypher parameter set."""
    assessment = commit.new_assessment
    plan = commit.new_plan
    question = commit.new_question
    acceptance = commit.plan_acceptance
    return {
        'user_uuid': user_uuid,
        'session_uuid': session_uuid,
        'turn_uuid': turn.uuid,
        'content': turn.content,
        'created_at': turn.created_at,
        'new_revision': expected_revision + 1,
        'phase': commit.phase.value,
        'new_assessment_uuid': assessment.uuid if assessment else None,
        'new_assessment_summary': assessment.summary if assessment else None,
        'new_assessment_strands_json': (
            json.dumps([strand.model_dump() for strand in assessment.strands])
            if assessment
            else None
        ),
        'new_assessment_created_at': assessment.created_at
        if assessment
        else None,
        'new_plan_uuid': plan.uuid if plan else None,
        'new_plan_summary': plan.summary if plan else None,
        'new_plan_created_at': plan.created_at if plan else None,
        'new_plan_steps': (
            [
                {
                    'uuid': step.uuid,
                    'title': step.title,
                    'objective': step.objective,
                }
                for step in plan.steps
            ]
            if plan
            else []
        ),
        'new_plan_dependencies': (
            [dependency.model_dump() for dependency in plan.dependencies]
            if plan
            else []
        ),
        'new_question_uuid': question.uuid if question else None,
        'new_question_kind': question.kind.value if question else None,
        'new_question_prompt': question.prompt if question else None,
        'new_question_response_json': (
            question.response.model_dump_json() if question else None
        ),
        'new_question_created_at': question.created_at if question else None,
        'new_question_step_uuid': commit.new_question_step_uuid,
        'current_assessment_uuid': commit.current_assessment_uuid,
        'current_plan_uuid': commit.current_plan_uuid,
        'current_step_uuid': commit.current_step_uuid,
        'pending_question_uuid': commit.pending_question_uuid,
        'acceptance_plan_uuid': acceptance.plan_uuid if acceptance else None,
        'acceptance_learner_turn_uuid': (
            acceptance.learner_turn_uuid if acceptance else None
        ),
    }


def _parse_snapshot(row: Mapping[str, Any]) -> TeachingSessionSnapshot:
    """Convert one aggregate Neo4j row to the private domain snapshot."""
    return TeachingSessionSnapshot(
        session=_parse_session(row),
        turns=[_parse_turn_record(turn) for turn in row['turns']],
        current_assessment=(
            _parse_assessment(row['current_assessment'])
            if row['current_assessment'] is not None
            else None
        ),
        current_plan=(
            _parse_plan(row['current_plan'])
            if row['current_plan'] is not None
            else None
        ),
        current_plan_acceptance=(
            TeachingPlanAcceptance.model_validate(
                row['current_plan_acceptance']
            )
            if row['current_plan_acceptance'] is not None
            else None
        ),
        current_step_uuid=row['current_step_uuid'],
        pending_question_uuid=row['pending_question_uuid'],
    )


def _parse_session(row: Mapping[str, Any]) -> TeachingSession:
    """Convert session properties from Neo4j temporal values."""
    return TeachingSession(
        uuid=row['uuid'],
        goal=row['goal'],
        status=TeachingStatus(row['status']),
        phase=row['phase'],
        revision=row['revision'],
        created_at=_native_datetime(row['created_at']),
        updated_at=_native_datetime(row['updated_at']),
        ended_at=(
            _native_datetime(row['ended_at'])
            if row['ended_at'] is not None
            else None
        ),
    )


def _parse_turn_record(row: Mapping[str, Any]) -> TeachingTurnRecord:
    """Convert one relationship-enriched turn row."""
    question = row['question']
    return TeachingTurnRecord(
        turn=TeachingTurn(
            uuid=row['turn']['uuid'],
            role=TeachingRole(row['turn']['role']),
            content=row['turn']['content'],
            created_at=_native_datetime(row['turn']['created_at']),
        ),
        question=_parse_question(question) if question is not None else None,
        question_step_uuid=row['question_step_uuid'],
        responds_to_question_uuid=row['responds_to_question_uuid'],
        presented_plan_uuid=row['presented_plan_uuid'],
        accepted_plan_uuid=row['accepted_plan_uuid'],
    )


def _parse_assessment(row: Mapping[str, Any]) -> ProbeAssessment:
    """Convert one persisted probe assessment."""
    strands = json.loads(row['strands_json'])
    return ProbeAssessment(
        uuid=row['uuid'],
        summary=row['summary'],
        strands=[ProbeStrand.model_validate(strand) for strand in strands],
        created_at=_native_datetime(row['created_at']),
    )


def _parse_plan(row: Mapping[str, Any]) -> TeachingPlan:
    """Convert one plan and order its steps prerequisite-first."""
    steps = [TeachingPlanStep.model_validate(step) for step in row['steps']]
    dependencies = [
        TeachingPlanDependency.model_validate(dependency)
        for dependency in row['dependencies']
    ]
    step_by_uuid = {step.uuid: step for step in steps}
    graph = {step_uuid: set() for step_uuid in step_by_uuid}
    for dependency in dependencies:
        graph[dependency.step_uuid].add(dependency.prerequisite_uuid)
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    ordered_steps: list[TeachingPlanStep] = []
    while sorter.is_active():
        ready = tuple(sorted(sorter.get_ready()))
        ordered_steps.extend(step_by_uuid[step_uuid] for step_uuid in ready)
        sorter.done(*ready)
    return TeachingPlan(
        uuid=row['uuid'],
        summary=row['summary'],
        created_at=_native_datetime(row['created_at']),
        steps=ordered_steps,
        dependencies=sorted(
            dependencies,
            key=lambda dependency: (
                dependency.step_uuid,
                dependency.prerequisite_uuid,
            ),
        ),
    )


def _parse_question(row: Mapping[str, Any]) -> TeachingQuestion:
    """Convert one private persisted question specification."""
    return TeachingQuestion(
        uuid=row['uuid'],
        kind=TeachingQuestionKind(row['kind']),
        prompt=row['prompt'],
        response=json.loads(row['response_json']),
        created_at=_native_datetime(row['created_at']),
    )


def _native_datetime(value: datetime | DateTime) -> datetime:
    """Convert a Neo4j temporal value to the domain datetime type."""
    return value.to_native() if isinstance(value, DateTime) else value
