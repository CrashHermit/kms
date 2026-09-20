"""DSPy module for deciding global procedure hub membership."""

import dspy

from kms2.core.model.global_semantic.procedure_hub import (
    GlobalProcedureHubJudgeDecision,
    GlobalProcedureHubJudgeInput,
)


class GlobalProcedureHubJudgeSignature(dspy.Signature):
    r"""Judge whether procedure descriptions express one canonical method.

    Return one boolean for each ordered pair. True means the two procedures may
    belong to the same final global procedure hub. False means do not connect
    them. Preserve ordered steps, conditions, inputs, outputs, and termination
    when comparing procedures.
    """

    requests: list[GlobalProcedureHubJudgeInput] = dspy.InputField(
        description='Ordered procedure pairs to judge independently.'
    )
    decisions: list[GlobalProcedureHubJudgeDecision] = dspy.OutputField(
        description='One indexed procedure-equivalence decision per pair.'
    )


class GlobalProcedureHubJudgeModule(dspy.Module):
    """Judge an ordered batch of procedure pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[GlobalProcedureHubJudgeInput]
    ) -> list[GlobalProcedureHubJudgeDecision]:
        """Judge procedure pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[GlobalProcedureHubJudgeInput]
    ) -> list[GlobalProcedureHubJudgeDecision]:
        """Judge procedure pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions


__all__ = ['GlobalProcedureHubJudgeModule', 'GlobalProcedureHubJudgeSignature']
