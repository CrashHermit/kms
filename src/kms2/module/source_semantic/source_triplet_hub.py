"""DSPy module for synthesizing source-local triplet hubs."""

import dspy

from kms2.core.model import (
    SourceTripletHubDefinition,
    SourceTripletHubSynthesisInput,
)


class SourceTripletHubSignature(dspy.Signature):
    r"""Synthesize one source-local assertion from fixed triplet evidence.

    Return a concise reusable canonical assertion and a standalone explanation
    grounded only in the supplied source facts and exact triplets. Preserve
    qualified, negative, conditional, quantified, and mathematical evidence.
    The evidence is fixed: never select membership, infer consequences, merge
    neighboring facts, or introduce unsupported facts.
    """

    request: SourceTripletHubSynthesisInput = dspy.InputField(
        description='Fixed source triplet evidence; do not decide membership.'
    )
    definition: SourceTripletHubDefinition = dspy.OutputField(
        description='One canonical source-local triplet hub definition.'
    )


class SourceTripletHubModule(dspy.Module):
    """Synthesize one source-local triplet hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletHubSynthesisInput
    ) -> SourceTripletHubDefinition:
        """Synthesize a triplet hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourceTripletHubSynthesisInput
    ) -> SourceTripletHubDefinition:
        """Synthesize a triplet hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourceTripletHubModule', 'SourceTripletHubSignature']
