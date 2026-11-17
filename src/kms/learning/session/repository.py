"""Owner-scoped Neo4j persistence for learning sessions."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession, AsyncTransaction
from pydantic import BaseModel

from kms.artifacts import ContentAddressedArtifactStore
from kms.concurrency import run_blocking_work
from kms.learning.session.models import (
    LearningDecisionRecord,
    LearningSession,
    SessionHistory,
)
from kms.learning.session.queries import (
    APPEND_STEP,
    APPEND_TEACHING_ASSOCIATIONS,
    ATTACH_DECISION_ASSOCIATIONS,
    ATTACH_TASK_ASSESSMENT,
    ATTACH_TASK_RESPONSE,
    CREATE_SESSION,
    CREATE_TASK_SOLUTION,
    LOCK_SESSION_FOR_APPEND,
    READ_OWNED_TASK,
    READ_OWNED_TASK_RESPONSE,
    READ_SESSION,
)
from kms.learning.session.records import (
    finalize_materials,
    hydrate_session,
    hydrate_task,
    hydrate_task_response,
)
from kms.learning.task.models import (
    Task,
    TaskAssessment,
    TaskResponse,
)
from kms.learning.teaching.models import Teaching


class LearningSessionRepository:
    """Persist session timelines and interaction records within user ownership."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        artifacts: ContentAddressedArtifactStore,
    ) -> None:
        self._session_factory = session_factory
        self._artifacts = artifacts

    async def create_session(
        self, user_uuid: str, session: LearningSession
    ) -> LearningSession:
        async with self._session_factory() as database_session:
            result = await database_session.run(
                CREATE_SESSION,
                user_uuid=user_uuid,
                uuid=session.uuid,
                goal=session.goal,
                targets=session.targets,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                invalidated_at=session.invalidated_at,
            )
            await result.single(strict=True)
        return session

    async def read_session(
        self, user_uuid: str, session_uuid: str
    ) -> SessionHistory:
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_SESSION, user_uuid=user_uuid, session_uuid=session_uuid
            )
            row = await result.single(strict=True)
        return hydrate_session(dict(row))

    async def append_task(
        self, user_uuid: str, session_uuid: str, task: Task
    ) -> Task:
        finalized = await self._finalize(task)
        payload = _payload(
            finalized,
            exclude={
                'uuid',
                'created_at',
                'updated_at',
                'invalidated_at',
                'target',
                'grounding',
                'solution',
            },
        )
        solution = finalized.solution
        solution_payload = _payload(
            solution,
            exclude={
                'uuid',
                'created_at',
                'updated_at',
                'invalidated_at',
                'grounding',
            },
        )
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_session(transaction, user_uuid, session_uuid)
                await self._append_step(
                    transaction,
                    user_uuid,
                    session_uuid,
                    finalized,
                    'Task',
                    payload,
                )
                grounding = finalized.grounding
                solution_grounding = solution.grounding
                result = await transaction.run(
                    CREATE_TASK_SOLUTION,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    task_uuid=finalized.uuid,
                    target_kind=finalized.target.kind,
                    target_uuid=finalized.target.uuid,
                    task_proposition_uuids=sorted(
                        set(grounding.proposition_uuids)
                    ),
                    task_argument_uuids=sorted(set(grounding.argument_uuids)),
                    task_unit_uuids=sorted(set(grounding.unit_uuids)),
                    solution_uuid=solution.uuid,
                    solution_payload_json=solution_payload,
                    solution_created_at=solution.created_at,
                    solution_updated_at=solution.updated_at,
                    solution_invalidated_at=solution.invalidated_at,
                    solution_proposition_uuids=sorted(
                        set(solution_grounding.proposition_uuids)
                    ),
                    solution_argument_uuids=sorted(
                        set(solution_grounding.argument_uuids)
                    ),
                    solution_unit_uuids=sorted(
                        set(solution_grounding.unit_uuids)
                    ),
                    created_at=finalized.created_at,
                )
                await result.single(strict=True)
        return finalized

    async def read_task(
        self, user_uuid: str, session_uuid: str, task_uuid: str
    ) -> Task:
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_OWNED_TASK,
                user_uuid=user_uuid,
                session_uuid=session_uuid,
                task_uuid=task_uuid,
            )
            row = await result.single(strict=True)
        return hydrate_task(dict(row))

    async def append_decision(
        self,
        user_uuid: str,
        session_uuid: str,
        decision: LearningDecisionRecord,
    ) -> LearningDecisionRecord:
        learning_decision = decision.decision
        assessment_uuids = sorted(set(learning_decision.assessment_uuids))
        target = getattr(learning_decision, 'target', None)
        payload = json.dumps(
            {
                'decision': learning_decision.model_dump(
                    mode='json', exclude={'assessment_uuids', 'target'}
                )
            },
            separators=(',', ':'),
        )
        is_finish = learning_decision.kind == 'finish'
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_session(transaction, user_uuid, session_uuid)
                await self._append_step(
                    transaction,
                    user_uuid,
                    session_uuid,
                    decision,
                    'LearningDecision',
                    payload,
                    is_finish=is_finish,
                )
                result = await transaction.run(
                    ATTACH_DECISION_ASSOCIATIONS,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    decision_uuid=decision.uuid,
                    assessment_uuids=assessment_uuids,
                    target_kind=target.kind if target is not None else None,
                    target_uuid=target.uuid if target is not None else None,
                    created_at=decision.created_at,
                )
                await result.single(strict=True)
        return decision

    async def append_teaching(
        self, user_uuid: str, session_uuid: str, teaching: Teaching
    ) -> Teaching:
        finalized = await self._finalize(teaching)
        payload = _payload(
            finalized,
            exclude={
                'uuid',
                'created_at',
                'updated_at',
                'invalidated_at',
                'target',
                'grounding',
                'assessment_uuids',
            },
        )
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_session(transaction, user_uuid, session_uuid)
                await self._append_step(
                    transaction,
                    user_uuid,
                    session_uuid,
                    finalized,
                    'Teaching',
                    payload,
                )
                result = await transaction.run(
                    APPEND_TEACHING_ASSOCIATIONS,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    teaching_uuid=finalized.uuid,
                    target_kind=finalized.target.kind,
                    target_uuid=finalized.target.uuid,
                    proposition_uuids=sorted(
                        set(finalized.grounding.proposition_uuids)
                    ),
                    argument_uuids=sorted(
                        set(finalized.grounding.argument_uuids)
                    ),
                    unit_uuids=sorted(set(finalized.grounding.unit_uuids)),
                    assessment_uuids=sorted(set(finalized.assessment_uuids)),
                    created_at=finalized.created_at,
                )
                await result.single(strict=True)
        return finalized

    async def record_task_response(
        self,
        user_uuid: str,
        session_uuid: str,
        response: TaskResponse,
    ) -> TaskResponse:
        finalized = await self._finalize(response)
        payload = _payload(
            finalized,
            exclude={
                'uuid',
                'created_at',
                'updated_at',
                'invalidated_at',
                'task_uuid',
            },
        )
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_session(transaction, user_uuid, session_uuid)
                result = await transaction.run(
                    ATTACH_TASK_RESPONSE,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    task_uuid=finalized.task_uuid,
                    uuid=finalized.uuid,
                    payload_json=payload,
                    created_at=finalized.created_at,
                    updated_at=finalized.updated_at,
                    invalidated_at=finalized.invalidated_at,
                )
                await result.single(strict=True)
        return finalized

    async def read_task_response(
        self, user_uuid: str, session_uuid: str, response_uuid: str
    ) -> TaskResponse:
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_OWNED_TASK_RESPONSE,
                user_uuid=user_uuid,
                session_uuid=session_uuid,
                response_uuid=response_uuid,
            )
            row = await result.single(strict=True)
        return hydrate_task_response(dict(row))

    async def record_task_assessment(
        self,
        user_uuid: str,
        session_uuid: str,
        assessment: TaskAssessment,
    ) -> TaskAssessment:
        payload = _payload(
            assessment,
            exclude={
                'uuid',
                'created_at',
                'updated_at',
                'invalidated_at',
                'response_uuid',
            },
        )
        async with self._session_factory() as database_session:
            async with (
                await database_session.begin_transaction() as transaction
            ):
                await self._lock_session(transaction, user_uuid, session_uuid)
                result = await transaction.run(
                    ATTACH_TASK_ASSESSMENT,
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    response_uuid=assessment.response_uuid,
                    uuid=assessment.uuid,
                    payload_json=payload,
                    created_at=assessment.created_at,
                    updated_at=assessment.updated_at,
                    invalidated_at=assessment.invalidated_at,
                )
                await result.single(strict=True)
        return assessment

    async def _finalize(self, record: BaseModel) -> BaseModel:
        return await run_blocking_work(
            finalize_materials, record, self._artifacts
        )

    @staticmethod
    async def _lock_session(
        transaction: AsyncTransaction, user_uuid: str, session_uuid: str
    ) -> dict[str, Any]:
        result = await transaction.run(
            LOCK_SESSION_FOR_APPEND,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
        )
        return await result.single(strict=True)

    @staticmethod
    async def _append_step(
        transaction: AsyncTransaction,
        user_uuid: str,
        session_uuid: str,
        record: Any,
        label: str,
        payload_json: str,
        *,
        is_finish: bool = False,
    ) -> None:
        query = APPEND_STEP.replace(':$step_label', f':{label}')
        result = await transaction.run(
            query,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
            uuid=record.uuid,
            payload_json=payload_json,
            created_at=record.created_at,
            updated_at=record.updated_at,
            invalidated_at=record.invalidated_at,
            is_finish=is_finish,
        )
        await result.single(strict=True)


def _payload(record: Any, *, exclude: set[str]) -> str:
    return json.dumps(
        record.model_dump(mode='json', exclude=exclude), separators=(',', ':')
    )
