"""DSPy module for deciding source-local procedure hub membership."""

import dspy

from kms2.core.model.semantic.source_procedure_hub import (
    SourceProcedureHubJudgeDecision,
    SourceProcedureHubJudgeInput,
)


class SourceProcedureHubJudgeSignature(dspy.Signature):
    r"""Judge whether procedure occurrences express one canonical method.

    Return one boolean for each ordered pair. True means the two occurrences
    may belong to the same final source-local procedure hub. False means do not
    connect them. Preserve ordered steps, conditions, inputs, outputs, and
    termination when comparing procedures.
    """

    requests: list[SourceProcedureHubJudgeInput] = dspy.InputField(
        description='Ordered procedure occurrence pairs to judge independently.'
    )
    decisions: list[SourceProcedureHubJudgeDecision] = dspy.OutputField(
        description='One indexed procedure-equivalence decision per pair.'
    )


class SourceProcedureHubJudgeModule(dspy.Module):
    """Judge an ordered batch of procedure occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourceProcedureHubJudgeInput]
    ) -> list[SourceProcedureHubJudgeDecision]:
        """Judge procedure pairs synchronously."""
        return self.predictor(requests=requests).decisions

    async def aforward(
        self, *, requests: list[SourceProcedureHubJudgeInput]
    ) -> list[SourceProcedureHubJudgeDecision]:
        """Judge procedure pairs asynchronously."""
        return (await self.predictor.acall(requests=requests)).decisions


__all__ = ['SourceProcedureHubJudgeModule', 'SourceProcedureHubJudgeSignature']
