"""DSPy module for synthesizing source-local predicate hubs."""

import dspy

from kms2.core.model import (
    SourcePredicateHubDefinition,
    SourcePredicateHubSynthesisInput,
)


class SourcePredicateHubSignature(dspy.Signature):
    r"""Synthesize one source-local relation from fixed predicate evidence.

    Return a concise predicate and source-grounded description for the supplied
    occurrences. Preserve directed relation semantics. The members are fixed
    evidence; never decide membership or invent a relation direction.
    """

    request: SourcePredicateHubSynthesisInput = dspy.InputField(
        description='Fixed predicate community evidence; do not decide membership.'
    )
    definition: SourcePredicateHubDefinition = dspy.OutputField(
        description='One canonical source-local predicate hub definition.'
    )


class SourcePredicateHubModule(dspy.Module):
    """Synthesize one source-local predicate hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateHubSynthesisInput
    ) -> SourcePredicateHubDefinition:
        """Synthesize a predicate hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: SourcePredicateHubSynthesisInput
    ) -> SourcePredicateHubDefinition:
        """Synthesize a predicate hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


__all__ = ['SourcePredicateHubModule', 'SourcePredicateHubSignature']
