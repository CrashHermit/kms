"""DSPy module for deciding source-local statement hub membership."""

import dspy

from kms2.core.model.semantic.source_statement_hub import (
    SourceStatementHubJudgeDecision,
    SourceStatementHubJudgeInput,
)


class SourceStatementHubJudgeSignature(dspy.Signature):
    r"""Judge whether statement occurrences express one canonical statement.

    Return one boolean for each ordered pair. True means the two occurrences
    may belong to the same final source-local statement hub. False means do not
    connect them. Reject topical relatedness and merely similar wording.
    """

    requests: list[SourceStatementHubJudgeInput] = dspy.InputField(
        description='Ordered statement occurrence pairs to judge independently.'
    )
    decisions: list[SourceStatementHubJudgeDecision] = dspy.OutputField(
        description='One indexed statement-equivalence decision per pair.'
    )


class SourceStatementHubJudgeModule(dspy.Module):
    """Judge an ordered batch of statement occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourceStatementHubJudgeInput]
    ) -> list[SourceStatementHubJudgeDecision]:
        """Judge statement pairs synchronously."""
        return self.predictor(requests=requests).decisions

    async def aforward(
        self, *, requests: list[SourceStatementHubJudgeInput]
    ) -> list[SourceStatementHubJudgeDecision]:
        """Judge statement pairs asynchronously."""
        return (await self.predictor.acall(requests=requests)).decisions


__all__ = ['SourceStatementHubJudgeModule', 'SourceStatementHubJudgeSignature']
