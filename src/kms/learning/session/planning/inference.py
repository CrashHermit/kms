"""Choose an assessment, teaching step, or finish for a learning session."""

import dspy

from kms.learning.session.models import LearningDecision
from kms.learning.session.planning.models import LearningPlanningRequest


class LearningPlanningSignature(dspy.Signature):
    """Choose the next learning action from the supplied session context.

    With empty history, assess one atomic understanding. Incorrect or unknown
    attempts call for targeted teaching, as does explicitly unsound reasoning
    even when the answer is correct. A correct answer without unnecessary
    explanation is uncertainty, not failure. After teaching, assess that same
    target before advancing or finishing. Use observations to select a missing
    connection instead of repeating all source material. Supported performance
    may advance to another relevant understanding or finish with a bounded,
    cited summary that states remaining uncertainty. Copy supplied or observed
    target identities exactly; never invent target UUIDs. A teaching decision
    always names an existing target. Do not retrieve knowledge, create teaching
    material, schedule reviews, use fixed turn limits, or route solely by a
    deterministic right/wrong rule.
    """

    request: LearningPlanningRequest = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decision: LearningDecision = dspy.OutputField()


class LearningPlanningModule(dspy.Module):
    """Run typed learning planning through an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: LearningPlanningRequest) -> LearningDecision:
        return self.predictor(request=request).decision

    async def aforward(
        self, *, request: LearningPlanningRequest
    ) -> LearningDecision:
        return (await self.predictor.acall(request=request)).decision
