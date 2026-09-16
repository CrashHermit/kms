"""DSPy module for synthesizing source-local event hubs."""

import dspy

from kms2.core.model import (
    SourceEventHubDefinition,
    SourceEventHubSynthesisInput,
)


class SourceEventHubSignature(dspy.Signature):
    r"""Synthesize one source-local event from fixed event evidence.

    Return a concise event name and source-grounded description for the supplied
    occurrences. The members are fixed evidence; never decide membership and do
    not rewrite the event as a consequence, result, or unrelated concept.
    """

    request: SourceEventHubSynthesisInput = dspy.InputField(
        description='Fixed event community evidence; do not decide membership.'
    )
    definition: SourceEventHubDefinition = dspy.OutputField(
        description='One canonical source-local event hub definition.'
    )


class SourceEventHubModule(dspy.Module):
    """Synthesize one source-local event hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventHubSynthesisInput
    ) -> SourceEventHubDefinition:
        """Synthesize an event hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourceEventHubSynthesisInput
    ) -> SourceEventHubDefinition:
        """Synthesize an event hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourceEventHubModule', 'SourceEventHubSignature']
