"""DSPy module for deciding source-local event hub membership."""

import dspy

from kms2.core.model.semantic.source_event_hub import (
    SourceEventHubJudgeDecision,
    SourceEventHubJudgeInput,
)


class SourceEventHubJudgeSignature(dspy.Signature):
    r"""Judge whether event occurrences describe the same occurrence or process.

    Return one boolean for each ordered pair. True means the two occurrences may
    belong to the same final source-local event hub. False means do not connect
    them. Reject topical relatedness, causes, consequences, and context events.
    """

    requests: list[SourceEventHubJudgeInput] = dspy.InputField(
        description='Ordered event occurrence pairs to judge independently.'
    )
    decisions: list[SourceEventHubJudgeDecision] = dspy.OutputField(
        description='One indexed same-occurrence decision per supplied pair.'
    )


class SourceEventHubJudgeModule(dspy.Module):
    """Judge an ordered batch of event occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourceEventHubJudgeInput]
    ) -> list[SourceEventHubJudgeDecision]:
        """Judge event pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[SourceEventHubJudgeInput]
    ) -> list[SourceEventHubJudgeDecision]:
        """Judge event pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['SourceEventHubJudgeModule', 'SourceEventHubJudgeSignature']
