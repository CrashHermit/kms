"""DSPy module for deciding global entity hub membership."""

import dspy

from kms2.core.model.global_semantic.entity_hub import (
    GlobalEntityHubJudgeDecision,
    GlobalEntityHubJudgeInput,
)


class GlobalEntityHubJudgeSignature(dspy.Signature):
    r"""Judge whether two entity source hubs identify the same canonical entity.

    Return one boolean for each ordered pair. True requires canonical identity
    equivalence. False means do not connect them. Reject topical relatedness,
    merely similar names, and lexical similarity without identity equivalence.
    """

    requests: list[GlobalEntityHubJudgeInput] = dspy.InputField(
        description=('Ordered entity pairs to judge independently.')
    )
    decisions: list[GlobalEntityHubJudgeDecision] = dspy.OutputField(
        description='One indexed entity decision per supplied pair.'
    )


class GlobalEntityHubJudgeModule(dspy.Module):
    """Judge an ordered batch of entity occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[GlobalEntityHubJudgeInput]
    ) -> list[GlobalEntityHubJudgeDecision]:
        """Judge entity pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[GlobalEntityHubJudgeInput]
    ) -> list[GlobalEntityHubJudgeDecision]:
        """Judge entity pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['GlobalEntityHubJudgeModule', 'GlobalEntityHubJudgeSignature']
