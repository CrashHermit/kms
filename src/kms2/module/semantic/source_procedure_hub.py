"""DSPy module for synthesizing source-local procedure hubs."""

import dspy

from kms2.core.model import (
    SourceProcedureHubDefinition,
    SourceProcedureHubSynthesisInput,
)


class SourceProcedureHubSignature(dspy.Signature):
    r"""Synthesize one source-local procedure from fixed evidence.

    Return a concise canonical name and source-grounded description for the
    supplied procedure occurrences. Preserve method order, steps, conditions,
    inputs, outputs, and termination. The members are fixed evidence; never
    decide membership or invent facts.
    """

    request: SourceProcedureHubSynthesisInput = dspy.InputField(
        description='Fixed procedure community evidence; do not decide membership.'
    )
    definition: SourceProcedureHubDefinition = dspy.OutputField(
        description='One canonical source-local procedure hub definition.'
    )


class SourceProcedureHubModule(dspy.Module):
    """Synthesize one source-local procedure hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceProcedureHubSynthesisInput
    ) -> SourceProcedureHubDefinition:
        """Synthesize a procedure hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourceProcedureHubSynthesisInput
    ) -> SourceProcedureHubDefinition:
        """Synthesize a procedure hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourceProcedureHubModule', 'SourceProcedureHubSignature']
