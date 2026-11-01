"""Transactional persistence for user-scoped teaching sessions."""

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from graphlib import TopologicalSorter
from typing import Any

from neo4j import AsyncSession, AsyncTransaction, Record

from kms2.core.model.knowledge.proposition import (
    KnowledgeAssertion,
    KnowledgeAssertionRecord,
)
from kms2.core.model.source import SourceRevisionReference
from kms2.core.model.source_processing.admission import StagedSourceSnapshot
from kms2.core.model.teaching.evidence import (
    TeachingCitationView,
    TeachingEvidenceReference,
)
from kms2.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
    TeachingPlanDependency,
    TeachingPlanStep,
)
from kms2.core.model.teaching.probe import ProbeAssessment, ProbeStrand
from kms2.core.model.teaching.session import (
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnCommit,
    TeachingTurnRecord,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.core.source_snapshot import assert_matching_snapshot
from kms2.database.knowledge.repository import write_knowledge_admission
from kms2.database.source.queries.source_snapshot import LOCK_SNAPSHOT_OWNER
from kms2.database.source.source_snapshot_repository import (
    create_owned_snapshot_in_transaction,
    read_owned_snapshot_in_transaction,
)
from kms2.database.teaching.observation_repository import (
    write_learner_observations,
)
from kms2.database.teaching.queries.evidence import (
    READ_TEACHING_CITATION_MEDIA,
    READ_TEACHING_CITATIONS,
)
from kms2.database.teaching.queries.observation import (
    AUTHORIZE_OWNED_LEARNER_OBSERVATIONS,
    LINK_ASSESSMENT_OBSERVATIONS,
)
from kms2.database.teaching.queries.session import (
    CHECK_OWNED_SOURCE_REVISION,
    CREATE_TEACHING_SESSION,
    END_TEACHING_SESSION,
    LOAD_TEACHING_SESSION,
    LOCK_OWNED_SOURCE,
    LOCK_TEACHING_SESSION,
    LOCK_TEACHING_USER_KNOWLEDGE,
    READ_TEACHING_SESSION_REVISION,
    READ_TEACHING_SESSIONS,
    READ_TEACHING_USER_KNOWLEDGE_REVISION,
)
from kms2.database.teaching.queries.turn import (
    APPEND_TEACHING_RESPONSE,
    CHECK_TEACHING_CITATIONS,
    RECORD_TEACHING_TURN,
)
from kms2.database.teaching.records import (
    learner_observation_from_record,
    native_datetime,
    question_from_record,
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

    async def read_teaching_citations(
        self,
        user_uuid: str,
        session_uuid: str,
        references: list[TeachingEvidenceReference],
    ) -> list[TeachingCitationView]:
        """Read only distinct evidence cited in the owned session."""
        unique = list(
            {
                (reference.kind, reference.uuid): reference
                for reference in references
            }.values()
        )
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_TEACHING_CITATIONS,
                user_uuid=user_uuid,
                session_uuid=session_uuid,
                references=[
                    {'kind': reference.kind, 'uuid': reference.uuid}
                    for reference in unique
                ],
            )
            row = await result.single(strict=True)
        by_key = {
            (citation['kind'], citation['uuid']): _parse_citation(citation)
            for citation in row['citations']
        }
        return [by_key[(item.kind, item.uuid)] for item in unique]

    async def read_teaching_citation_media_path(
        self,
        user_uuid: str,
        session_uuid: str,
        asset_uuid: str,
    ) -> str:
        """Authorize an image citation and return its private storage path."""
        async with self._session_factory() as database_session:
            result = await database_session.run(
                READ_TEACHING_CITATION_MEDIA,
                user_uuid=user_uuid,
                session_uuid=session_uuid,
                asset_uuid=asset_uuid,
            )
            row = await result.single(strict=True)
        return row['path']

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
                if commit.staged_sources:
                    await self._single(
                        transaction,
                        LOCK_SNAPSHOT_OWNER,
                        user_uuid=user_uuid,
                    )
                if commit.knowledge is not None:
                    await self._single(
                        transaction,
                        LOCK_TEACHING_USER_KNOWLEDGE,
                        user_uuid=user_uuid,
                    )
                    await self._single(
                        transaction,
                        READ_TEACHING_USER_KNOWLEDGE_REVISION,
                        user_uuid=user_uuid,
                        expected_revision=commit.knowledge.expected_revision,
                    )

                staged_by_uuid: dict[str, StagedSourceSnapshot] = {}
                for candidate in commit.staged_sources:
                    previous = staged_by_uuid.get(candidate.source.uuid)
                    if previous is not None:
                        assert_matching_snapshot(candidate, previous)
                    else:
                        staged_by_uuid[candidate.source.uuid] = candidate
                source_revisions = [
                    *commit.source_revisions,
                    *(
                        commit.knowledge.source_revisions
                        if commit.knowledge is not None
                        else []
                    ),
                ]
                source_revisions.extend(
                    SourceRevisionReference(
                        source_uuid=candidate.source.uuid,
                        revision=candidate.source.revision,
                    )
                    for candidate in staged_by_uuid.values()
                )
                requested_revisions = {
                    (item.source_uuid, item.revision): item
                    for item in source_revisions
                }
                source_uuids = sorted(
                    {source_uuid for source_uuid, _ in requested_revisions}
                    | set(staged_by_uuid)
                )
                for source_uuid in source_uuids:
                    candidate = staged_by_uuid.get(source_uuid)
                    if candidate is not None:
                        await create_owned_snapshot_in_transaction(
                            transaction,
                            user_uuid,
                            candidate.source,
                            candidate.pages,
                        )
                    await self._single(
                        transaction,
                        LOCK_OWNED_SOURCE,
                        user_uuid=user_uuid,
                        source_uuid=source_uuid,
                    )
                    if candidate is not None:
                        persisted = await read_owned_snapshot_in_transaction(
                            transaction,
                            user_uuid,
                            source_uuid,
                        )
                        assert_matching_snapshot(candidate, persisted)
                for reference in sorted(
                    requested_revisions.values(),
                    key=lambda item: (item.source_uuid, item.revision),
                ):
                    await self._single(
                        transaction,
                        CHECK_OWNED_SOURCE_REVISION,
                        user_uuid=user_uuid,
                        source_uuid=reference.source_uuid,
                        revision=reference.revision,
                    )
                parameters = _record_parameters(
                    user_uuid=user_uuid,
                    session_uuid=session_uuid,
                    expected_revision=expected_revision,
                    turn=turn,
                    commit=commit,
                )
                await self._single(
                    transaction,
                    CHECK_TEACHING_CITATIONS,
                    user_uuid=user_uuid,
                    citation_block_uuids=parameters['citation_block_uuids'],
                    citation_image_uuids=parameters['citation_image_uuids'],
                    citation_proposition_uuids=parameters[
                        'citation_proposition_uuids'
                    ],
                    supporting_block_uuids=parameters['supporting_block_uuids'],
                    source_revisions=parameters['source_revisions'],
                )
                existing_observation_uuids = sorted(
                    set(
                        commit.new_assessment.observation_uuids
                        if commit.new_assessment is not None
                        else []
                    )
                    - {
                        item.observation.uuid
                        for item in commit.learner_observations
                    }
                )
                if existing_observation_uuids:
                    await self._single(
                        transaction,
                        AUTHORIZE_OWNED_LEARNER_OBSERVATIONS,
                        user_uuid=user_uuid,
                        observation_uuids=existing_observation_uuids,
                    )
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
                    **parameters,
                )
                await write_learner_observations(
                    transaction,
                    user_uuid,
                    turn.uuid,
                    commit.learner_observations,
                )
                if commit.new_assessment is not None:
                    observation_uuids = sorted(
                        set(commit.new_assessment.observation_uuids)
                    )
                    if observation_uuids:
                        await self._single(
                            transaction,
                            LINK_ASSESSMENT_OBSERVATIONS,
                            user_uuid=user_uuid,
                            session_uuid=session_uuid,
                            assessment_uuid=commit.new_assessment.uuid,
                            observation_uuids=observation_uuids,
                        )
                if commit.knowledge is not None:
                    await write_knowledge_admission(
                        transaction,
                        user_uuid,
                        turn.uuid,
                        commit.knowledge,
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
    citation_pairs = sorted(
        {(reference.kind, reference.uuid) for reference in commit.citations}
    )
    supporting_block_uuids = sorted(
        {
            block_uuid
            for assertion in commit.knowledge.assertions
            for block_uuid in assertion.supporting_block_uuids
        }
        | {
            block_uuid
            for step in commit.knowledge.new_reasoning_steps
            for block_uuid in step.supporting_block_uuids
        }
        if commit.knowledge is not None
        else set()
    )
    citation_block_uuids = sorted(
        {uuid for kind, uuid in citation_pairs if kind == 'block'}
    )
    citation_image_uuids = [
        uuid for kind, uuid in citation_pairs if kind == 'image'
    ]
    citation_proposition_uuids = [
        uuid for kind, uuid in citation_pairs if kind == 'proposition'
    ]
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
        'workspace_json': commit.workspace_json,
        'acceptance_plan_uuid': acceptance.plan_uuid if acceptance else None,
        'acceptance_learner_turn_uuid': (
            acceptance.learner_turn_uuid if acceptance else None
        ),
        'citation_block_uuids': citation_block_uuids,
        'supporting_block_uuids': supporting_block_uuids,
        'citation_image_uuids': citation_image_uuids,
        'source_revisions': [
            {'source_uuid': item.source_uuid, 'revision': item.revision}
            for item in sorted(
                {
                    (item.source_uuid, item.revision): item
                    for item in (
                        commit.source_revisions
                        + (
                            commit.knowledge.source_revisions
                            if commit.knowledge is not None
                            else []
                        )
                    )
                }.values(),
                key=lambda item: (item.source_uuid, item.revision),
            )
        ],
        'citation_proposition_uuids': citation_proposition_uuids,
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
        workspace_json=row['workspace_json'],
    )


def _parse_session(row: Mapping[str, Any]) -> TeachingSession:
    """Convert session properties from Neo4j temporal values."""
    return TeachingSession(
        uuid=row['uuid'],
        goal=row['goal'],
        status=TeachingStatus(row['status']),
        phase=row['phase'],
        revision=row['revision'],
        created_at=native_datetime(row['created_at']),
        updated_at=native_datetime(row['updated_at']),
        ended_at=(
            native_datetime(row['ended_at'])
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
            created_at=native_datetime(row['turn']['created_at']),
        ),
        question=question_from_record(question)
        if question is not None
        else None,
        question_step_uuid=row['question_step_uuid'],
        responds_to_question_uuid=row['responds_to_question_uuid'],
        presented_plan_uuid=row['presented_plan_uuid'],
        accepted_plan_uuid=row['accepted_plan_uuid'],
        citations=[
            TeachingEvidenceReference(kind=kind, uuid=uuid)
            for kind, uuid in sorted(
                {
                    (reference['kind'], reference['uuid'])
                    for reference in row['citations']
                }
            )
        ],
        knowledge_assertions=[
            _parse_knowledge_assertion(item)
            for item in row['knowledge_assertions']
        ],
        learner_observations=[
            learner_observation_from_record(item)
            for item in row['learner_observations']
        ],
    )


def _parse_knowledge_assertion(
    row: Mapping[str, Any],
) -> KnowledgeAssertionRecord:
    """Convert one per-turn assertion and its one-hop references."""
    return KnowledgeAssertionRecord(
        assertion=KnowledgeAssertion(
            uuid=row['uuid'],
            created_at=native_datetime(row['created_at']),
        ),
        proposition_uuid=row['proposition_uuid'],
        supporting_block_uuids=sorted(row['supporting_block_uuids']),
        reasoning_step_uuids=sorted(row['reasoning_step_uuids']),
        teaching_turn_uuid=row['teaching_turn_uuid'],
    )


def _parse_citation(row: Mapping[str, Any]) -> TeachingCitationView:
    """Convert a session-authorized citation projection."""
    metadata = json.loads(row['metadata']) if row.get('metadata') else {}
    retrieved_at = metadata.get('retrieved_at')
    if isinstance(retrieved_at, str):
        retrieved_at = datetime.fromisoformat(retrieved_at)
    return TeachingCitationView(
        kind=row['kind'],
        uuid=row['uuid'],
        content=row['content'],
        source_uuid=row.get('source_uuid'),
        source_key=row.get('source_key'),
        url=metadata.get('url'),
        title=metadata.get('title'),
        retrieved_at=(
            native_datetime(retrieved_at) if retrieved_at is not None else None
        ),
        page_indexes=sorted(
            index for index in row.get('page_indexes', []) if index is not None
        ),
        image_uuids=sorted(
            uuid for uuid in row.get('image_uuids', []) if uuid is not None
        ),
    )


def _parse_assessment(row: Mapping[str, Any]) -> ProbeAssessment:
    """Convert one persisted probe assessment."""
    strands = json.loads(row['strands_json'])
    return ProbeAssessment(
        uuid=row['uuid'],
        summary=row['summary'],
        strands=[ProbeStrand.model_validate(strand) for strand in strands],
        observation_uuids=sorted(row['observation_uuids']),
        created_at=native_datetime(row['created_at']),
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
        created_at=native_datetime(row['created_at']),
        steps=ordered_steps,
        dependencies=sorted(
            dependencies,
            key=lambda dependency: (
                dependency.step_uuid,
                dependency.prerequisite_uuid,
            ),
        ),
    )
