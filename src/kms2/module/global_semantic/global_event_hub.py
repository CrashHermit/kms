"""DSPy module for synthesizing global event hubs."""

import dspy

from kms2.core.model.global_semantic.global_event_hub import (
    GlobalEventHubDefinition,
    GlobalEventHubSummary,
    GlobalEventHubSummaryInput,
    GlobalEventHubSummaryMergeInput,
    GlobalEventHubSummarySynthesisInput,
    GlobalEventHubSynthesisInput,
)


class GlobalEventHubSignature(dspy.Signature):
    r"""Synthesize a canonical global event from fixed evidence.

    Accept either original source-hub evidence or ordered partial summaries.
    Return a concise canonical event name and description. The supplied
    evidence is fixed; never decide membership. Do not treat evidence field
    labels as domain claims. Describe the occurrence rather than its
    consequence.
    """

    request: (
        GlobalEventHubSynthesisInput | GlobalEventHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed event evidence or ordered partial summaries; do not decide membership.'
    )
    definition: GlobalEventHubDefinition = dspy.OutputField(
        description='One canonical global event hub definition.'
    )


class GlobalEventHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied global event evidence.

    Preserve member and field associations, distinguishing content, conditions,
    negation, quantities, and mathematics. Remove repetition and emit a concise,
    self-contained partial summary. Do not choose a canonical hub name or
    change membership.
    """

    request: GlobalEventHubSummaryInput = dspy.InputField(
        description='Ordered event evidence records to summarize.'
    )
    summary: GlobalEventHubSummary = dspy.OutputField(
        description='One temporary partial event summary.'
    )


class GlobalEventHubSummaryMergeSignature(dspy.Signature):
    r"""Consolidate ordered partial global event summaries.

    Remove repetition while preserving distinctions, qualifications, and
    member/field associations. Do not infer agreement or consequences. Produce
    one concise temporary summary, not a hub definition.
    """

    request: GlobalEventHubSummaryMergeInput = dspy.InputField(
        description='Ordered event partial summaries to consolidate.'
    )
    summary: GlobalEventHubSummary = dspy.OutputField(
        description='One consolidated temporary event summary.'
    )


class GlobalEventHubModule(dspy.Module):
    """Synthesize one global event hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: GlobalEventHubSynthesisInput
        | GlobalEventHubSummarySynthesisInput,
    ) -> GlobalEventHubDefinition:
        """Synthesize an event hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: GlobalEventHubSynthesisInput
        | GlobalEventHubSummarySynthesisInput,
    ) -> GlobalEventHubDefinition:
        """Synthesize an event hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


class GlobalEventHubSummaryModule(dspy.Module):
    """Summarize global event evidence in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEventHubSummaryInput
    ) -> GlobalEventHubSummary:
        """Summarize event evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalEventHubSummaryInput
    ) -> GlobalEventHubSummary:
        """Summarize event evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalEventHubSummaryMergeModule(dspy.Module):
    """Merge global event summaries in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEventHubSummaryMergeInput
    ) -> GlobalEventHubSummary:
        """Merge event summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalEventHubSummaryMergeInput
    ) -> GlobalEventHubSummary:
        """Merge event summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary
