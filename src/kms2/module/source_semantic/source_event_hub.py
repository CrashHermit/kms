"""DSPy modules for synthesizing source-local event hubs."""

import dspy

from kms2.core.model.source_semantic.source_event_hub import (
    SourceEventHubDefinition,
    SourceEventHubSummary,
    SourceEventHubSummaryInput,
    SourceEventHubSummaryMergeInput,
    SourceEventHubSummarySynthesisInput,
    SourceEventHubSynthesisInput,
)


class SourceEventHubSignature(dspy.Signature):
    r"""Synthesize one source-local event from fixed event evidence.

    Return a concise event name and source-grounded description for the supplied
    occurrences or ordered temporary summaries. The evidence is fixed; never
    decide membership and do not rewrite the event as a consequence, result, or
    unrelated concept.
    """

    request: (
        SourceEventHubSynthesisInput | SourceEventHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed event evidence or reduced summaries; do not decide membership.'
    )
    definition: SourceEventHubDefinition = dspy.OutputField(
        description='One canonical source-local event hub definition.'
    )


class SourceEventHubSummarySignature(dspy.Signature):
    r"""Summarize ordered event evidence into a concise, self-contained summary.

    Summarize only the supplied evidence, preserving distinguishing content,
    conditions, negation, quantities, mathematics, and direction while removing
    repetition. Keep each member's name and description associated. Do not
    choose a canonical hub name or change membership.
    """

    request: SourceEventHubSummaryInput = dspy.InputField(
        description='Ordered event evidence to summarize.'
    )
    summary: SourceEventHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary event evidence summary.'
    )


class SourceEventHubSummaryModule(dspy.Module):
    """Summarize source-local event evidence in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventHubSummaryInput
    ) -> SourceEventHubSummary:
        """Summarize event evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceEventHubSummaryInput
    ) -> SourceEventHubSummary:
        """Summarize event evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceEventHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered temporary event summaries without inventing agreement.

    Consolidate repetition while preserving distinctions, qualifications, and
    member associations. Do not infer agreement or consequences. Produce one
    concise, self-contained temporary summary, not a hub definition.
    """

    request: SourceEventHubSummaryMergeInput = dspy.InputField(
        description='Two or more ordered temporary event summaries.'
    )
    summary: SourceEventHubSummary = dspy.OutputField(
        description='One merged temporary event evidence summary.'
    )


class SourceEventHubSummaryMergeModule(dspy.Module):
    """Merge source-local event summaries in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventHubSummaryMergeInput
    ) -> SourceEventHubSummary:
        """Merge event summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceEventHubSummaryMergeInput
    ) -> SourceEventHubSummary:
        """Merge event summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceEventHubModule(dspy.Module):
    """Synthesize one source-local event hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: SourceEventHubSynthesisInput
        | SourceEventHubSummarySynthesisInput,
    ) -> SourceEventHubDefinition:
        """Synthesize an event hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourceEventHubSynthesisInput
        | SourceEventHubSummarySynthesisInput,
    ) -> SourceEventHubDefinition:
        """Synthesize an event hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
