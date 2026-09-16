"""DSPy module for deciding source-local entity hub membership."""

import dspy

from kms2.core.model.semantic.source_entity_hub import (
    SourceEntityHubJudgeDecision,
    SourceEntityHubJudgeInput,
)


class SourceEntityHubJudgeSignature(dspy.Signature):
    r"""Judge whether entity occurrences denote one canonical identity.

    Return one boolean for each ordered pair. True means the two occurrences may
    belong to the same final source-local entity hub. False means do not connect
    them. Reject topical relatedness, co-occurrence, and merely similar names.
    """

    requests: list[SourceEntityHubJudgeInput] = dspy.InputField(
        description='Ordered entity occurrence pairs to judge independently.'
    )
    decisions: list[SourceEntityHubJudgeDecision] = dspy.OutputField(
        description='One indexed identity-equivalence decision per supplied pair.'
    )


class SourceEntityHubJudgeModule(dspy.Module):
    """Judge an ordered batch of entity occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourceEntityHubJudgeInput]
    ) -> list[SourceEntityHubJudgeDecision]:
        """Judge entity pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[SourceEntityHubJudgeInput]
    ) -> list[SourceEntityHubJudgeDecision]:
        """Judge entity pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['SourceEntityHubJudgeModule', 'SourceEntityHubJudgeSignature']
