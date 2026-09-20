"""DSPy module for deciding global event hub membership."""

import dspy

from kms2.core.model.global_semantic.event_hub import (
    GlobalEventHubJudgeDecision,
    GlobalEventHubJudgeInput,
)


class GlobalEventHubJudgeSignature(dspy.Signature):
    r"""Judge whether event occurrences describe the same occurrence or process.

    Return one boolean for each ordered pair. True means the two events are
    the same occurrence or process and may belong to the same final global
    event hub. False means do not connect them. Reject causes, consequences,
    context events, and topical relatedness.
    """

    requests: list[GlobalEventHubJudgeInput] = dspy.InputField(
        description='Ordered event pairs to judge independently.'
    )
    decisions: list[GlobalEventHubJudgeDecision] = dspy.OutputField(
        description='One indexed event decision per supplied pair.'
    )


class GlobalEventHubJudgeModule(dspy.Module):
    """Judge an ordered batch of event occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[GlobalEventHubJudgeInput]
    ) -> list[GlobalEventHubJudgeDecision]:
        """Judge event pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[GlobalEventHubJudgeInput]
    ) -> list[GlobalEventHubJudgeDecision]:
        """Judge event pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['GlobalEventHubJudgeModule', 'GlobalEventHubJudgeSignature']
