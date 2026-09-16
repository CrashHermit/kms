"""DSPy module for synthesizing source-local entity hubs."""

import dspy

from kms2.core.model import (
    SourceEntityHubDefinition,
    SourceEntityHubSynthesisInput,
)


class SourceEntityHubSignature(dspy.Signature):
    r"""Synthesize one source-local concept from fixed entity evidence.

    Return a concise canonical name and source-grounded description for the
    supplied entity occurrences. The members are fixed evidence; never decide
    membership, add aliases from outside the members, or invent facts.
    """

    request: SourceEntityHubSynthesisInput = dspy.InputField(
        description='Fixed entity community evidence; do not decide membership.'
    )
    definition: SourceEntityHubDefinition = dspy.OutputField(
        description='One canonical source-local entity hub definition.'
    )


class SourceEntityHubModule(dspy.Module):
    """Synthesize one source-local entity hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityHubSynthesisInput
    ) -> SourceEntityHubDefinition:
        """Synthesize an entity hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourceEntityHubSynthesisInput
    ) -> SourceEntityHubDefinition:
        """Synthesize an entity hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourceEntityHubModule', 'SourceEntityHubSignature']
