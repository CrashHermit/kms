"""Interpret a learner's goal from the clarification exchange."""

import dspy

from kms.learning.probe.goal.models import GoalClarificationInput, GoalDecision


class GoalClarificationSignature(dspy.Signature):
    """Determine whether the learner's goal is specific enough for focused retrieval.

    Use the complete ordered exchange. If a missing detail would materially
    change the retrieval goal, ask one focused clarification question. Otherwise
    return a concise, faithful goal preserving the learner's scope and constraints.
    Do not assess mastery, teach, invent prerequisites, generate search queries,
    or require clarification when the goal is already specific enough.
    """

    request: GoalClarificationInput = dspy.InputField(
        description='Ordered learner and assistant clarification messages.'
    )
    decision: GoalDecision = dspy.OutputField(
        description='A clarification question or the clarified learning goal.'
    )


class GoalClarificationModule(dspy.Module):
    """Interpret goal clarification through an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: GoalClarificationInput) -> GoalDecision:
        """Interpret an exchange synchronously."""
        return self.predictor(request=request).decision

    async def aforward(
        self, *, request: GoalClarificationInput
    ) -> GoalDecision:
        """Interpret an exchange asynchronously."""
        prediction = await self.predictor.acall(request=request)
        return prediction.decision
