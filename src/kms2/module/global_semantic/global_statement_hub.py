"""DSPy module for synthesizing global statement hubs."""

import dspy

from kms2.core.model import (
    GlobalStatementHubDefinition,
    GlobalStatementHubSynthesisInput,
)


class GlobalStatementHubSignature(dspy.Signature):
    r"""Synthesize one cross-source statement from fixed evidence.

    Return a concise canonical name and description for the supplied source-hub
    members. Preserve whether the statement is a claim, fact, theorem,
    explanation, question, or exercise. The members are fixed evidence; never
    decide membership or invent evidence.
    """

    request: GlobalStatementHubSynthesisInput = dspy.InputField(
        description=(
            'Fixed statement community evidence; preserve statement qualifiers '
            'and do not decide membership.'
        )
    )
    definition: GlobalStatementHubDefinition = dspy.OutputField(
        description='One canonical global statement hub definition.'
    )


class GlobalStatementHubModule(dspy.Module):
    """Synthesize one global statement hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalStatementHubSynthesisInput
    ) -> GlobalStatementHubDefinition:
        """Synthesize a statement hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalStatementHubSynthesisInput
    ) -> GlobalStatementHubDefinition:
        """Synthesize a statement hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['GlobalStatementHubModule', 'GlobalStatementHubSignature']
