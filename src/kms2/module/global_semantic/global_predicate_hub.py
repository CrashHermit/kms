"""DSPy module for synthesizing global predicate hubs."""

import dspy

from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHubDefinition,
    GlobalPredicateHubSynthesisInput,
)


class GlobalPredicateHubSignature(dspy.Signature):
    r"""Synthesize one cross-source predicate relation from fixed evidence.

    Return a concise canonical predicate and description for the supplied
    source-hub members. The members are fixed evidence; never decide
    membership or use directed subject/object context.
    """

    request: GlobalPredicateHubSynthesisInput = dspy.InputField(
        description='Fixed predicate community evidence; do not decide membership.'
    )
    definition: GlobalPredicateHubDefinition = dspy.OutputField(
        description='One canonical global predicate hub definition.'
    )


class GlobalPredicateHubModule(dspy.Module):
    """Synthesize one global predicate hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalPredicateHubSynthesisInput
    ) -> GlobalPredicateHubDefinition:
        """Synthesize a predicate hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalPredicateHubSynthesisInput
    ) -> GlobalPredicateHubDefinition:
        """Synthesize a predicate hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
