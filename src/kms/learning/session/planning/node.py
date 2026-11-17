"""Persist the next model-selected learning decision."""

from kms.learning.session.models import LearningDecisionRecord
from kms.learning.session.planning.inference import LearningPlanningModule
from kms.learning.session.planning.models import PlanningState
from kms.learning.session.repository import LearningSessionRepository


class LearningPlanningNode:
    """Plan and persist one next step for an owned learning session."""

    def __init__(
        self,
        module: LearningPlanningModule,
        repository: LearningSessionRepository,
    ) -> None:
        self._module = module
        self._repository = repository

    async def run(
        self, state: PlanningState
    ) -> dict[str, LearningDecisionRecord]:
        await self._repository.read_session(
            state['user_uuid'], state['session_uuid']
        )
        decision = await self._module.acall(request=state['request'])
        record = LearningDecisionRecord(decision=decision)
        persisted = await self._repository.append_decision(
            state['user_uuid'], state['session_uuid'], record
        )
        return {'decision': persisted}
