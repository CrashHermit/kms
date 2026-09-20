"""DSPy module for deciding global statement hub membership."""

import dspy

from kms2.core.model.global_semantic.statement_hub import (
    GlobalStatementHubJudgeDecision,
    GlobalStatementHubJudgeInput,
)


class GlobalStatementHubJudgeSignature(dspy.Signature):
    r"""Judge whether two statement descriptions express the same statement.

    Return one boolean for each ordered pair. True means the two descriptions
    may belong to the same final global statement hub. Preserve whether each
    is a claim, fact, theorem, explanation, question, or exercise. Reject
    merely related wording, lexical overlap, and topical relatedness.
    """

    requests: list[GlobalStatementHubJudgeInput] = dspy.InputField(
        description='Ordered statement description pairs to judge independently.'
    )
    decisions: list[GlobalStatementHubJudgeDecision] = dspy.OutputField(
        description='One indexed statement decision per supplied pair.'
    )


class GlobalStatementHubJudgeModule(dspy.Module):
    """Judge an ordered batch of statement description pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[GlobalStatementHubJudgeInput]
    ) -> list[GlobalStatementHubJudgeDecision]:
        """Judge statement pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[GlobalStatementHubJudgeInput]
    ) -> list[GlobalStatementHubJudgeDecision]:
        """Judge statement pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['GlobalStatementHubJudgeModule', 'GlobalStatementHubJudgeSignature']
