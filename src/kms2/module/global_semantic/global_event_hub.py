"""DSPy module for synthesizing global event hubs."""

import dspy

from kms2.core.model import (
    GlobalEventHubDefinition,
    GlobalEventHubSynthesisInput,
)


class GlobalEventHubSignature(dspy.Signature):
    r"""Synthesize one cross-source event from fixed evidence.

    Return a concise canonical event name and description for the supplied
    source-hub members. The members are fixed evidence; never decide
    membership or use directed subject/object context.
    """

    request: GlobalEventHubSynthesisInput = dspy.InputField(
        description='Fixed event community evidence; do not decide membership.'
    )
    definition: GlobalEventHubDefinition = dspy.OutputField(
        description='One canonical global event hub definition.'
    )


class GlobalEventHubModule(dspy.Module):
    """Synthesize one global event hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEventHubSynthesisInput
    ) -> GlobalEventHubDefinition:
        """Synthesize an event hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalEventHubSynthesisInput
    ) -> GlobalEventHubDefinition:
        """Synthesize an event hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['GlobalEventHubModule', 'GlobalEventHubSignature']
