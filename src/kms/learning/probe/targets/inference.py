"""Decompose a clarified learning goal into atomic understandings."""

import dspy

from kms.learning.probe.goal.models import ClarifiedGoal


class TargetDecompositionSignature(dspy.Signature):
    """Decompose the learner's clarified goal into atomic target understandings.

    Return an ordered list of distinct learning needs required to satisfy the
    goal. Split a goal only when it asks for multiple independently useful
    understandings; keep a single request as one target even when it contains
    examples, values, assumptions, or context. Examples illustrate a learning
    need and must not become standalone targets.

    Each target must be independently understandable without the original
    conversation. Preserve relevant audience, setting, scope, constraints,
    assumptions, and misconceptions from the goal rather than replacing them
    with generic wording. Include directly relevant prerequisites only when
    needed for the stated goal; do not expand into an exhaustive curriculum or
    unrelated topics. Express targets as things to understand, not asserted
    propositions, answers, or learner quiz questions. Do not assume what the
    learner already knows.
    """

    goal: ClarifiedGoal = dspy.InputField()
    targets: list[str] = dspy.OutputField()


class TargetDecompositionModule(dspy.Module):
    """Produce atomic targets with an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, goal: ClarifiedGoal) -> list[str]:
        return self.predictor(goal=goal).targets

    async def aforward(self, *, goal: ClarifiedGoal) -> list[str]:
        return (await self.predictor.acall(goal=goal)).targets
