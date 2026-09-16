"""DSPy module for synthesizing source-local statement hubs."""

import dspy

from kms2.core.model import (
    SourceStatementHubDefinition,
    SourceStatementHubSynthesisInput,
)


class SourceStatementHubSignature(dspy.Signature):
    r"""Synthesize one source-local statement from fixed evidence.

    Return a concise canonical name and source-grounded description for the
    supplied statement occurrences. Preserve claim, fact, theorem, explanation,
    question, and exercise qualifiers. The members are fixed evidence; never
    decide membership or invent facts.
    """

    request: SourceStatementHubSynthesisInput = dspy.InputField(
        description='Fixed statement community evidence; do not decide membership.'
    )
    definition: SourceStatementHubDefinition = dspy.OutputField(
        description='One canonical source-local statement hub definition.'
    )


class SourceStatementHubModule(dspy.Module):
    """Synthesize one source-local statement hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceStatementHubSynthesisInput
    ) -> SourceStatementHubDefinition:
        """Synthesize a statement hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourceStatementHubSynthesisInput
    ) -> SourceStatementHubDefinition:
        """Synthesize a statement hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourceStatementHubModule', 'SourceStatementHubSignature']
