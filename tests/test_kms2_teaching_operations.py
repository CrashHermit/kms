from datetime import UTC, datetime

from kms2.core.model.teaching.observation import (
    ExistingLearnerObservationReference,
    LearnerObservation,
    LearnerObservationKind,
    LearnerObservationRecord,
    LearnerObservationWrite,
    NewLearnerObservationReference,
)
from kms2.core.model.teaching.operation import (
    AcceptPlan,
    ContinueProbing,
    Hint,
    ProposePlan,
    ReopenProbing,
    RespondAction,
    TeachingAssessmentDraft,
    TeachingOperation,
    TeachingPlanDraft,
    TeachingPlanStepDraft,
    TeachingQuestionDraft,
)
from kms2.core.model.teaching.plan import TeachingPlan
from kms2.core.model.teaching.probe import ProbeAssessment
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
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
from kms2.node.teaching.operations import prepare_instructional_commit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _snapshot(
    *,
    phase: TeachingPhase = TeachingPhase.PROBING,
    plan: TeachingPlan | None = None,
    current_step_uuid: str | None = None,
    current_assessment: ProbeAssessment | None = None,
    turns: list[TeachingTurnRecord] | None = None,
    pending_question_uuid: str | None = 'pending-question',
) -> TeachingSessionSnapshot:
    return TeachingSessionSnapshot(
        session=TeachingSession(
            uuid='session-1',
            goal='Understand the topic',
            status=TeachingStatus.AWAITING_TEACHER,
            phase=phase,
            revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        turns=turns or [],
        current_assessment=current_assessment,
        current_plan=plan,
        current_plan_acceptance=None,
        current_step_uuid=current_step_uuid,
        pending_question_uuid=pending_question_uuid,
    )


def _question(kind: TeachingQuestionKind, prompt: str) -> TeachingQuestionDraft:
    return TeachingQuestionDraft(
        kind=kind,
        prompt=prompt,
        response=FreeResponseSpecification(reference_answer='answer'),
    )


def _response(operation: TeachingOperation) -> RespondAction:
    return RespondAction(
        content='Response',
        content_kind='coordination',
        operation=operation,
    )


def test_continue_probing_replaces_question_and_assessment_provenance():
    old_observation = LearnerObservation(
        uuid='old-observation',
        kind=LearnerObservationKind.GAP,
        topic='topic',
        description='Earlier gap',
        embedding=[0.1],
        created_at=_NOW,
    )
    new_observation = LearnerObservation(
        uuid='new-observation',
        kind=LearnerObservationKind.MISCONCEPTION,
        topic='topic',
        description='Current misconception',
        embedding=[0.2],
        created_at=_NOW,
    )
    assessment = TeachingAssessmentDraft(
        summary='Current understanding',
        strands=[],
        observation_references=[
            ExistingLearnerObservationReference(kind='existing', position=0),
            NewLearnerObservationReference(kind='new', position=0),
        ],
    )
    commit = prepare_instructional_commit(
        _snapshot(),
        _response(
            ContinueProbing(
                kind='continue_probing',
                question=_question(TeachingQuestionKind.PROBE, 'Another probe'),
                assessment=assessment,
            )
        ),
        visible_turn_indexes=[],
        visible_observation_indexes=[0],
        learner_history=[
            LearnerObservationRecord(observation=old_observation, evidence=[])
        ],
        prepared_observations=[
            LearnerObservationWrite(
                observation=new_observation,
                evidence_turn_uuids=['learner-turn'],
            )
        ],
    )

    assert commit.phase is TeachingPhase.PROBING
    assert commit.current_assessment_uuid == commit.new_assessment.uuid
    assert commit.new_assessment.observation_uuids == [
        'old-observation',
        'new-observation',
    ]
    assert commit.new_question.uuid != 'pending-question'
    assert commit.pending_question_uuid == commit.new_question.uuid
    assert commit.new_question_step_uuid is None
    assert commit.current_plan_uuid is None


def test_propose_then_accept_plan_links_the_existing_learner_acceptance():
    learner = TeachingTurn(
        uuid='learner-acceptance',
        role=TeachingRole.LEARNER,
        content='I accept the plan.',
        created_at=_NOW,
    )
    turn_record = TeachingTurnRecord(
        turn=learner,
        question=None,
        question_step_uuid=None,
        responds_to_question_uuid='approval-question',
        presented_plan_uuid=None,
        accepted_plan_uuid=None,
    )
    proposal = prepare_instructional_commit(
        _snapshot(turns=[turn_record]),
        _response(
            ProposePlan(
                kind='propose_plan',
                plan=TeachingPlanDraft(
                    summary='Two steps',
                    steps=[
                        TeachingPlanStepDraft(
                            title='First',
                            objective='Start',
                            prerequisite_positions=[],
                        ),
                        TeachingPlanStepDraft(
                            title='Second',
                            objective='Continue',
                            prerequisite_positions=[0],
                        ),
                    ],
                ),
                question=_question(
                    TeachingQuestionKind.PLAN_APPROVAL,
                    'Do you accept this plan?',
                ),
            )
        ),
        visible_turn_indexes=[0],
        visible_observation_indexes=[],
        learner_history=[],
        prepared_observations=[],
    )

    assert proposal.phase is TeachingPhase.PLANNING
    assert proposal.current_plan_uuid == proposal.new_plan.uuid
    assert proposal.current_step_uuid is None
    assert (
        proposal.new_plan.dependencies[0].prerequisite_uuid
        == proposal.new_plan.steps[0].uuid
    )
    assert proposal.new_question_step_uuid is None

    accepted = prepare_instructional_commit(
        _snapshot(
            phase=TeachingPhase.PLANNING,
            plan=proposal.new_plan,
            pending_question_uuid=proposal.new_question.uuid,
            turns=[turn_record],
        ),
        _response(
            AcceptPlan(
                kind='accept_plan', learner_turn_position=0, step_position=1
            )
        ),
        visible_turn_indexes=[0],
        visible_observation_indexes=[],
        learner_history=[],
        prepared_observations=[],
    )

    assert accepted.phase is TeachingPhase.TEACHING
    assert accepted.current_plan_uuid == proposal.new_plan.uuid
    assert accepted.current_step_uuid == proposal.new_plan.steps[1].uuid
    assert accepted.pending_question_uuid is None
    assert accepted.plan_acceptance.plan_uuid == proposal.new_plan.uuid
    assert accepted.plan_acceptance.learner_turn_uuid == learner.uuid


def test_hint_retains_exact_heads_and_reopening_clears_active_plan():
    plan = TeachingPlan(
        uuid='plan-1',
        summary='A plan',
        created_at=_NOW,
        steps=[],
        dependencies=[],
    )
    previous_assessment = ProbeAssessment(
        uuid='assessment-1',
        summary='Previous assessment',
        strands=[],
        created_at=_NOW,
    )
    snapshot = _snapshot(
        phase=TeachingPhase.TEACHING,
        plan=plan,
        current_step_uuid='step-1',
        pending_question_uuid='check-1',
        current_assessment=previous_assessment,
    )
    common = {
        'visible_turn_indexes': [],
        'visible_observation_indexes': [],
        'learner_history': [],
        'prepared_observations': [],
    }
    hinted = prepare_instructional_commit(
        snapshot,
        _response(Hint(kind='hint')),
        **common,
    )
    assert hinted.phase is TeachingPhase.TEACHING
    assert hinted.current_plan_uuid == 'plan-1'
    assert hinted.current_step_uuid == 'step-1'
    assert hinted.pending_question_uuid == 'check-1'
    assert hinted.current_assessment_uuid == previous_assessment.uuid

    reopened = prepare_instructional_commit(
        snapshot,
        _response(
            ReopenProbing(
                kind='reopen_probing',
                question=_question(TeachingQuestionKind.PROBE, 'Start again'),
            )
        ),
        **common,
    )
    assert reopened.phase is TeachingPhase.PROBING
    assert reopened.current_plan_uuid is None
    assert reopened.current_step_uuid is None
    assert reopened.pending_question_uuid == reopened.new_question.uuid
