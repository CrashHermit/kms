"""DSPy module for synthesizing source-local predicate hubs."""

import dspy

from kms2.core.model.source_semantic.source_predicate_hub import (
    SourcePredicateHubDefinition,
    SourcePredicateHubSummary,
    SourcePredicateHubSummaryInput,
    SourcePredicateHubSummaryMergeInput,
    SourcePredicateHubSummarySynthesisInput,
    SourcePredicateHubSynthesisInput,
)


class SourcePredicateHubSummarySignature(dspy.Signature):
    r"""Summarize ordered source evidence for a predicate hub.

    Preserve predicate direction and distinguishing qualifications. Emit a
    concise, self-contained summary of the supplied evidence.
    """

    request: SourcePredicateHubSummaryInput = dspy.InputField(
        description='Ordered predicate evidence to summarize.'
    )
    summary: SourcePredicateHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary predicate-hub evidence summary.'
    )


class SourcePredicateHubSummaryModule(dspy.Module):
    """Generate one partial predicate-hub summary."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateHubSummaryInput
    ) -> SourcePredicateHubSummary:
        """Generate a predicate summary synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourcePredicateHubSummaryInput
    ) -> SourcePredicateHubSummary:
        """Generate a predicate summary asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourcePredicateHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered predicate summaries without changing their meaning.

    Preserve qualifications and their associations without inferring agreement
    or consequences. Return one concise, self-contained temporary summary.
    """

    request: SourcePredicateHubSummaryMergeInput = dspy.InputField(
        description='Ordered predicate summaries to consolidate.'
    )
    summary: SourcePredicateHubSummary = dspy.OutputField(
        description='One temporary merged predicate-hub summary.'
    )


class SourcePredicateHubSummaryMergeModule(dspy.Module):
    """Merge partial predicate-hub summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateHubSummaryMergeInput
    ) -> SourcePredicateHubSummary:
        """Merge predicate summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourcePredicateHubSummaryMergeInput
    ) -> SourcePredicateHubSummary:
        """Merge predicate summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourcePredicateHubSignature(dspy.Signature):
    r"""Synthesize one source-local relation from fixed predicate evidence.

    Return a concise predicate and source-grounded description. Preserve
    directed relation semantics. The evidence is fixed; never decide membership
    or invent a relation direction.
    """

    request: (
        SourcePredicateHubSynthesisInput
        | SourcePredicateHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed predicate evidence or ordered partial summaries.'
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
        self,
        *,
        request: SourcePredicateHubSynthesisInput
        | SourcePredicateHubSummarySynthesisInput,
    ) -> SourcePredicateHubDefinition:
        """Synthesize a predicate hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourcePredicateHubSynthesisInput
        | SourcePredicateHubSummarySynthesisInput,
    ) -> SourcePredicateHubDefinition:
        """Synthesize a predicate hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
