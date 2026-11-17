"""Adaptive planning persists one decision against the owned session."""

import asyncio

from kms.learning.references import PropositionTarget
from kms.learning.session.models import (
    AssessmentTarget,
    LearningDecision,
    LearningDecisionRecord,
    SessionContext,
)
from kms.learning.session.planning.models import LearningPlanningRequest
from kms.learning.session.planning.node import LearningPlanningNode


def test_planner_reads_owner_before_inference_and_returns_persisted_record() -> (
    None
):
    events: list[str] = []
    selected = AssessmentTarget(
        understanding='Distinguish function value from limit.',
        explanation='Start with one atomic target.',
        assessment_uuids=[],
        target=PropositionTarget(uuid='target'),
    )
    persisted = LearningDecisionRecord(
        uuid='persisted-decision', decision=selected
    )

    class Repository:
        async def read_session(self, user_uuid: str, session_uuid: str) -> None:
            assert (user_uuid, session_uuid) == ('user', 'session')
            events.append('read')

        async def append_decision(
            self,
            user_uuid: str,
            session_uuid: str,
            decision: LearningDecisionRecord,
        ) -> LearningDecisionRecord:
            assert events == ['read', 'infer']
            assert decision.decision == selected
            events.append('persist')
            return persisted

    class Module:
        async def acall(
            self, *, request: LearningPlanningRequest
        ) -> LearningDecision:
            assert request.session.goal == 'Understand limits.'
            assert request.target == selected.target
            events.append('infer')
            return selected

    state = {
        'user_uuid': 'user',
        'session_uuid': 'session',
        'request': LearningPlanningRequest(
            session=SessionContext(
                goal='Understand limits.',
                targets=['Distinguish function value from limit.'],
                steps=[],
                responses=[],
                assessments=[],
            ),
            target=selected.target,
        ),
    }
    result = asyncio.run(
        LearningPlanningNode(Module(), Repository()).run(state)
    )

    assert events == ['read', 'infer', 'persist']
    assert result['decision'] is persisted
    assert result['decision'].uuid == 'persisted-decision'
