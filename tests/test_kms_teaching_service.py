import asyncio
from copy import deepcopy
from datetime import UTC, datetime

from kms.core.model.teaching.evidence import TeachingEvidenceReference
from kms.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
    TeachingPlanStep,
)
from kms.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestion,
    TeachingQuestionKind,
)
from kms.core.model.teaching.session import (
    TeachingPhase,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnCommit,
    TeachingTurnRecord,
)
from kms.core.model.teaching.turn import TeachingRole
from kms.service.teaching import TeachingService


class _MemoryTeachingRepository:
    def __init__(self):
        self.snapshots = {}
        self.plans = {}

    async def create_session(self, user_uuid, session):
        snapshot = TeachingSessionSnapshot(
            session=session,
            turns=[],
            current_assessment=None,
            current_plan=None,
            current_plan_acceptance=None,
            current_step_uuid=None,
            pending_question_uuid=None,
        )
        self.snapshots[(user_uuid, session.uuid)] = snapshot
        return deepcopy(snapshot)

    async def list_sessions(self, user_uuid):
        return [
            deepcopy(snapshot.session)
            for (owner, _), snapshot in self.snapshots.items()
            if owner == user_uuid
        ]

    async def load_session(self, user_uuid, session_uuid):
        return deepcopy(self.snapshots[(user_uuid, session_uuid)])

    async def append_response(
        self, user_uuid, session_uuid, expected_revision, turn
    ):
        snapshot = self.snapshots[(user_uuid, session_uuid)]
        assert snapshot.session.revision == expected_revision
        assert snapshot.session.status is TeachingStatus.AWAITING_LEARNER
        snapshot.turns.append(
            TeachingTurnRecord(
                turn=turn,
                question=None,
                question_step_uuid=None,
                responds_to_question_uuid=snapshot.pending_question_uuid,
                presented_plan_uuid=None,
                accepted_plan_uuid=None,
            )
        )
        snapshot.session.status = TeachingStatus.AWAITING_TEACHER
        snapshot.session.revision += 1
        return deepcopy(snapshot)

    async def record_turn(
        self, user_uuid, session_uuid, expected_revision, turn, commit
    ):
        snapshot = self.snapshots[(user_uuid, session_uuid)]
        assert snapshot.session.revision == expected_revision
        assert snapshot.session.status is TeachingStatus.AWAITING_TEACHER
        if commit.new_plan is not None:
            self.plans[commit.new_plan.uuid] = commit.new_plan
        snapshot.turns.append(
            TeachingTurnRecord(
                turn=turn,
                question=commit.new_question,
                question_step_uuid=commit.new_question_step_uuid,
                responds_to_question_uuid=None,
                presented_plan_uuid=(
                    commit.new_plan.uuid
                    if commit.new_plan is not None
                    else None
                ),
                accepted_plan_uuid=None,
                citations=deepcopy(commit.citations),
            )
        )
        if commit.plan_acceptance is not None:
            for record in snapshot.turns:
                if record.turn.uuid == commit.plan_acceptance.learner_turn_uuid:
                    record.accepted_plan_uuid = commit.plan_acceptance.plan_uuid
        if commit.new_plan is not None:
            snapshot.current_plan = commit.new_plan
        snapshot.current_plan_acceptance = commit.plan_acceptance
        snapshot.current_step_uuid = commit.current_step_uuid
        snapshot.pending_question_uuid = commit.pending_question_uuid
        snapshot.session.phase = commit.phase
        snapshot.session.status = TeachingStatus.AWAITING_LEARNER
        snapshot.session.revision += 1
        return deepcopy(snapshot)

    async def end_session(
        self, user_uuid, session_uuid, expected_revision, ended_at
    ):
        snapshot = self.snapshots[(user_uuid, session_uuid)]
        assert snapshot.session.revision == expected_revision
        snapshot.session.status = TeachingStatus.ENDED
        snapshot.session.ended_at = ended_at
        snapshot.session.updated_at = ended_at
        snapshot.session.revision += 1
        snapshot.pending_question_uuid = None
        return deepcopy(snapshot)


def _plan(uuid: str, step_uuid: str) -> TeachingPlan:
    return TeachingPlan(
        uuid=uuid,
        summary=uuid,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        steps=[
            TeachingPlanStep(uuid=step_uuid, title='Step', objective='Learn')
        ],
        dependencies=[],
    )


def _commit(
    *,
    plan: TeachingPlan | None,
    question: TeachingQuestion | None,
    plan_acceptance: TeachingPlanAcceptance | None = None,
    phase=TeachingPhase.PROBING,
    current_step_uuid=None,
    pending_question_uuid=None,
    citations: list[TeachingEvidenceReference] | None = None,
):
    return TeachingTurnCommit(
        phase=phase,
        current_assessment_uuid=None,
        current_plan_uuid=plan.uuid if plan is not None else None,
        current_step_uuid=current_step_uuid,
        pending_question_uuid=pending_question_uuid,
        source_revisions=[],
        new_plan=plan,
        new_question=question,
        new_question_step_uuid=current_step_uuid,
        plan_acceptance=plan_acceptance,
        citations=citations or [],
    )


def test_service_transitions_preserve_pending_questions_and_plan_acceptance():
    async def exercise():
        repository = _MemoryTeachingRepository()
        service = TeachingService(repository)
        started = await service.start_session('user-1', 'Understand slope')
        snapshot = await service.load_snapshot(
            'user-1',
            started.session.uuid,
        )
        assert snapshot.session.uuid == started.session.uuid
        question = TeachingQuestion(
            kind=TeachingQuestionKind.PROBE,
            prompt='What is slope?',
            response=FreeResponseSpecification(
                reference_answer='rise over run'
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        plan = _plan('plan-1', 'step-1')
        first = await service.record_result(
            'user-1',
            started.session.uuid,
            'Let us probe your understanding.',
            _commit(
                plan=plan,
                question=question,
                current_step_uuid='step-1',
                pending_question_uuid=question.uuid,
                citations=[
                    TeachingEvidenceReference(kind='block', uuid='block-1')
                ],
            ),
            expected_revision=0,
        )
        assert first.turns[-1].turn.role is TeachingRole.TEACHER
        assert first.turns[-1].citations == [
            TeachingEvidenceReference(kind='block', uuid='block-1')
        ]
        after_response = await service.submit_response(
            'user-1',
            started.session.uuid,
            'Rise over run.',
            expected_revision=1,
        )
        assert after_response.session.status is TeachingStatus.AWAITING_TEACHER
        assert after_response.pending_question_uuid == question.uuid
        assert (
            after_response.turns[-1].responds_to_question_uuid == question.uuid
        )

        accepted = await service.record_teaching_turn(
            'user-1',
            started.session.uuid,
            'The plan is accepted.',
            _commit(
                plan=None,
                question=None,
                plan_acceptance=TeachingPlanAcceptance(
                    plan_uuid=plan.uuid,
                    learner_turn_uuid=after_response.turns[-1].turn.uuid,
                ),
                phase=TeachingPhase.TEACHING,
                current_step_uuid='step-1',
                pending_question_uuid=None,
            ),
            expected_revision=2,
        )
        assert accepted.current_plan_acceptance.plan_uuid == plan.uuid
        assert accepted.turns[-1 - 1].accepted_plan_uuid == plan.uuid
        assert accepted.current_plan.uuid == plan.uuid
        revision_after_plan_response = await service.submit_response(
            'user-1',
            started.session.uuid,
            'I accept the plan.',
            expected_revision=3,
        )
        assert (
            revision_after_plan_response.turns[-1].responds_to_question_uuid
            is None
        )

        replacement_question = TeachingQuestion(
            kind=TeachingQuestionKind.CLARIFICATION,
            prompt='What part needs clarification?',
            response=FreeResponseSpecification(),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        replacement_plan = _plan('plan-2', 'step-2')
        replacement = await service.record_teaching_turn(
            'user-1',
            started.session.uuid,
            'We will revise the plan.',
            _commit(
                plan=replacement_plan,
                question=replacement_question,
                phase=TeachingPhase.PLANNING,
                current_step_uuid='step-2',
                pending_question_uuid=replacement_question.uuid,
            ),
            expected_revision=4,
        )
        assert replacement.current_plan.uuid == 'plan-2'
        assert replacement.current_plan_acceptance is None
        assert [
            record.question.uuid
            for record in replacement.turns
            if record.question
        ] == [
            question.uuid,
            replacement_question.uuid,
        ]

    asyncio.run(exercise())
