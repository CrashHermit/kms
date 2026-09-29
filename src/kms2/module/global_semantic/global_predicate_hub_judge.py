"""DSPy module for deciding global predicate hub membership."""

import dspy

from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHubJudgeDecision,
    GlobalPredicateHubJudgeInput,
)


class GlobalPredicateHubJudgeSignature(dspy.Signature):
    r"""Judge whether predicate occurrences express one relation.

    Return one boolean for each ordered pair. True means the two relations may
    belong to the same final global predicate hub. False means do not
    connect them. Reject lexical or topical similarity.
    """

    requests: list[GlobalPredicateHubJudgeInput] = dspy.InputField(
        description=('Ordered predicate pairs to judge independently.')
    )
    decisions: list[GlobalPredicateHubJudgeDecision] = dspy.OutputField(
        description='One indexed predicate decision per supplied pair.'
    )


class GlobalPredicateHubJudgeModule(dspy.Module):
    """Judge an ordered batch of predicate occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[GlobalPredicateHubJudgeInput]
    ) -> list[GlobalPredicateHubJudgeDecision]:
        """Judge predicate pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[GlobalPredicateHubJudgeInput]
    ) -> list[GlobalPredicateHubJudgeDecision]:
        """Judge predicate pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions
