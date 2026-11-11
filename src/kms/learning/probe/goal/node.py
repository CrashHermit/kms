"""Interpret the current learning-goal exchange."""

from kms.learning.probe.goal.inference import GoalClarificationModule
from kms.learning.probe.goal.models import GoalClarificationInput, GoalDecision
from kms.learning.probe.goal.state import GoalClarificationState


class GoalClarificationNode:
    """Decide whether the learner's goal needs clarification."""

    def __init__(self, module: GoalClarificationModule) -> None:
        self._module = module

    async def run(
        self, state: GoalClarificationState
    ) -> dict[str, GoalDecision]:
        """Interpret the complete local exchange."""
        decision = await self._module.acall(
            request=GoalClarificationInput(messages=state.messages)
        )
        return {'decision': decision}
