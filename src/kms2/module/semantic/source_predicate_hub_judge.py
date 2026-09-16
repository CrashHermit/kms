"""DSPy module for deciding source-local predicate hub membership."""

import dspy

from kms2.core.model.semantic.source_predicate_hub import (
    SourcePredicateHubJudgeDecision,
    SourcePredicateHubJudgeInput,
)


class SourcePredicateHubJudgeSignature(dspy.Signature):
    r"""Judge whether directed relation occurrences express one relation.

    Return one boolean for each ordered pair. True means the two relations may
    belong to the same final source-local predicate hub. False means do not
    connect them. Preserve direction and reject lexical or topical similarity.
    """

    requests: list[SourcePredicateHubJudgeInput] = dspy.InputField(
        description=(
            'Ordered subject-predicate-object relation pairs to judge independently.'
        )
    )
    decisions: list[SourcePredicateHubJudgeDecision] = dspy.OutputField(
        description='One indexed directed-relation decision per supplied pair.'
    )


class SourcePredicateHubJudgeModule(dspy.Module):
    """Judge an ordered batch of predicate occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourcePredicateHubJudgeInput]
    ) -> list[SourcePredicateHubJudgeDecision]:
        """Judge predicate pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[SourcePredicateHubJudgeInput]
    ) -> list[SourcePredicateHubJudgeDecision]:
        """Judge predicate pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['SourcePredicateHubJudgeModule', 'SourcePredicateHubJudgeSignature']
