"""DSPy module for synthesizing global entity hubs."""

import dspy

from kms2.core.model import (
    GlobalEntityHubDefinition,
    GlobalEntityHubSynthesisInput,
)


class GlobalEntityHubSignature(dspy.Signature):
    r"""Synthesize one cross-source entity from fixed source-hub evidence.

    Return a concise canonical entity name and description for the supplied
    source-hub members. The members are fixed evidence; never decide
    membership or invent evidence.
    """

    request: GlobalEntityHubSynthesisInput = dspy.InputField(
        description='Fixed entity community evidence; do not decide membership.'
    )
    definition: GlobalEntityHubDefinition = dspy.OutputField(
        description='One canonical global entity hub definition.'
    )


class GlobalEntityHubModule(dspy.Module):
    """Synthesize one global entity hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEntityHubSynthesisInput
    ) -> GlobalEntityHubDefinition:
        """Synthesize an entity hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalEntityHubSynthesisInput
    ) -> GlobalEntityHubDefinition:
        """Synthesize an entity hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['GlobalEntityHubModule', 'GlobalEntityHubSignature']
