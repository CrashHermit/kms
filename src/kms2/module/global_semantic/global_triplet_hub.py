"""DSPy module for synthesizing global triplet hubs."""

import dspy

from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHubDefinition,
    GlobalTripletHubSynthesisInput,
)


class GlobalTripletHubSignature(dspy.Signature):
    r"""Synthesize one cross-source relation from fixed triplet evidence.

    Return a concise reusable canonical relation and a standalone explanation
    grounded only in the supplied global roles and source-triplet-hub evidence.
    The evidence is fixed: never select membership, infer consequences, merge
    neighboring relations, or introduce unsupported facts.
    """

    request: GlobalTripletHubSynthesisInput = dspy.InputField(
        description='Fixed global triplet evidence; do not decide membership.'
    )
    definition: GlobalTripletHubDefinition = dspy.OutputField(
        description='One canonical global triplet hub definition.'
    )


class GlobalTripletHubModule(dspy.Module):
    """Synthesize one global triplet hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalTripletHubSynthesisInput
    ) -> GlobalTripletHubDefinition:
        """Synthesize a global triplet hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalTripletHubSynthesisInput
    ) -> GlobalTripletHubDefinition:
        """Synthesize a global triplet hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
