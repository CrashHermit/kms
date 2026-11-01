"""Translate instructional operations into complete teaching-turn commits."""

from datetime import UTC, datetime
from typing import assert_never

from kms2.core.model.teaching.observation import (
    LearnerObservationRecord,
    LearnerObservationWrite,
)
from kms2.core.model.teaching.operation import (
    AcceptPlan,
    Clarify,
    CompleteProbing,
    Consolidate,
    ContinueProbing,
    Hint,
    ProposePlan,
    ReopenProbing,
    RepairCheck,
    RespondAction,
    ReturnToPlanning,
    TeachingAssessmentDraft,
    TeachingPlanDraft,
    TeachingQuestionDraft,
    TeachStep,
)
from kms2.core.model.teaching.plan import (
    TeachingPlan,
    TeachingPlanAcceptance,
    TeachingPlanDependency,
    TeachingPlanStep,
)
from kms2.core.model.teaching.probe import ProbeAssessment
from kms2.core.model.teaching.question import TeachingQuestion
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingSessionSnapshot,
    TeachingTurnCommit,
)


def prepare_instructional_commit(
    snapshot: TeachingSessionSnapshot,
    response: RespondAction,
    *,
    visible_turn_indexes: list[int],
    visible_observation_indexes: list[int],
    learner_history: list[LearnerObservationRecord],
    prepared_observations: list[LearnerObservationWrite],
) -> TeachingTurnCommit:
    """Build the complete intended state for one typed instructional operation."""
    operation = response.operation
    created_at = datetime.now(UTC)
    current_assessment_uuid = (
        snapshot.current_assessment.uuid
        if snapshot.current_assessment is not None
        else None
    )
    current_plan = snapshot.current_plan
    current_step_uuid = snapshot.current_step_uuid
    pending_question_uuid = snapshot.pending_question_uuid
    new_assessment = None
    new_plan = None
    new_question = None
    new_question_step_uuid = None
    plan_acceptance = None
    phase = snapshot.session.phase

    if isinstance(operation, ContinueProbing):
        phase = TeachingPhase.PROBING
        new_question = _new_question(operation.question, created_at)
        if operation.assessment is not None:
            new_assessment = _new_assessment(
                operation.assessment,
                created_at,
                visible_observation_indexes,
                learner_history,
                prepared_observations,
            )
    elif isinstance(operation, CompleteProbing):
        phase = TeachingPhase.PLANNING
        new_assessment = _new_assessment(
            operation.assessment,
            created_at,
            visible_observation_indexes,
            learner_history,
            prepared_observations,
        )
        pending_question_uuid = None
    elif isinstance(operation, ProposePlan):
        phase = TeachingPhase.PLANNING
        new_plan = _new_plan(operation.plan, created_at)
        current_plan = new_plan
        current_step_uuid = None
        new_question = _new_question(operation.question, created_at)
    elif isinstance(operation, AcceptPlan):
        phase = TeachingPhase.TEACHING
        current_step_uuid = current_plan.steps[operation.step_position].uuid
        learner_turn_index = visible_turn_indexes[
            operation.learner_turn_position
        ]
        learner_turn = snapshot.turns[learner_turn_index].turn
        plan_acceptance = TeachingPlanAcceptance(
            plan_uuid=current_plan.uuid,
            learner_turn_uuid=learner_turn.uuid,
        )
        pending_question_uuid = None
    elif isinstance(operation, TeachStep):
        phase = TeachingPhase.TEACHING
        current_step_uuid = current_plan.steps[operation.step_position].uuid
        new_question = _new_question(operation.question, created_at)
        new_question_step_uuid = current_step_uuid
        if operation.assessment is not None:
            new_assessment = _new_assessment(
                operation.assessment,
                created_at,
                visible_observation_indexes,
                learner_history,
                prepared_observations,
            )
    elif isinstance(operation, RepairCheck):
        phase = TeachingPhase.TEACHING
        new_question = _new_question(operation.question, created_at)
        new_question_step_uuid = current_step_uuid
        if operation.assessment is not None:
            new_assessment = _new_assessment(
                operation.assessment,
                created_at,
                visible_observation_indexes,
                learner_history,
                prepared_observations,
            )
    elif isinstance(operation, Hint):
        pass
    elif isinstance(operation, Consolidate):
        phase = TeachingPhase.TEACHING
        pending_question_uuid = None
        if operation.assessment is not None:
            new_assessment = _new_assessment(
                operation.assessment,
                created_at,
                visible_observation_indexes,
                learner_history,
                prepared_observations,
            )
    elif isinstance(operation, ReopenProbing):
        phase = TeachingPhase.PROBING
        current_plan = None
        current_step_uuid = None
        new_question = _new_question(operation.question, created_at)
        if operation.assessment is not None:
            new_assessment = _new_assessment(
                operation.assessment,
                created_at,
                visible_observation_indexes,
                learner_history,
                prepared_observations,
            )
    elif isinstance(operation, ReturnToPlanning):
        phase = TeachingPhase.PLANNING
        current_plan = None
        current_step_uuid = None
        if operation.question is None:
            pending_question_uuid = None
        else:
            new_question = _new_question(operation.question, created_at)
    elif isinstance(operation, Clarify):
        if operation.question is not None:
            new_question = _new_question(operation.question, created_at)
            new_question_step_uuid = current_step_uuid
    else:
        assert_never(operation)

    if new_assessment is not None:
        current_assessment_uuid = new_assessment.uuid
    return TeachingTurnCommit(
        phase=phase,
        current_assessment_uuid=current_assessment_uuid,
        current_plan_uuid=current_plan.uuid
        if current_plan is not None
        else None,
        current_step_uuid=current_step_uuid,
        pending_question_uuid=(
            new_question.uuid
            if new_question is not None
            else pending_question_uuid
        ),
        source_revisions=[],
        new_assessment=new_assessment,
        new_plan=new_plan,
        new_question=new_question,
        new_question_step_uuid=new_question_step_uuid,
        plan_acceptance=plan_acceptance,
    )


def _new_plan(draft: TeachingPlanDraft, created_at: datetime) -> TeachingPlan:
    """Create one immutable plan revision and dependency graph."""
    steps = [
        TeachingPlanStep(title=step.title, objective=step.objective)
        for step in draft.steps
    ]
    dependencies = [
        TeachingPlanDependency(
            step_uuid=steps[step_position].uuid,
            prerequisite_uuid=steps[prerequisite_position].uuid,
        )
        for step_position, step in enumerate(draft.steps)
        for prerequisite_position in step.prerequisite_positions
    ]
    return TeachingPlan(
        summary=draft.summary,
        created_at=created_at,
        steps=steps,
        dependencies=dependencies,
    )


def _new_question(
    draft: TeachingQuestionDraft, created_at: datetime
) -> TeachingQuestion:
    """Create one immutable question from its model-authored draft."""
    return TeachingQuestion(
        kind=draft.kind,
        prompt=draft.prompt,
        response=draft.response,
        created_at=created_at,
    )


def _new_assessment(
    draft: TeachingAssessmentDraft,
    created_at: datetime,
    visible_observation_indexes: list[int],
    learner_history: list[LearnerObservationRecord],
    prepared_observations: list[LearnerObservationWrite],
) -> ProbeAssessment:
    """Create an assessment with exactly the observations it references."""
    observation_uuids = []
    for reference in draft.observation_references:
        if reference.kind == 'existing':
            history_index = visible_observation_indexes[reference.position]
            observation_uuids.append(
                learner_history[history_index].observation.uuid
            )
        else:
            observation_uuids.append(
                prepared_observations[reference.position].observation.uuid
            )
    return ProbeAssessment(
        summary=draft.summary,
        strands=draft.strands,
        observation_uuids=observation_uuids,
        created_at=created_at,
    )
