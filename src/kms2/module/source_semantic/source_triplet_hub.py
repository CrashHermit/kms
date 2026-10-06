"""DSPy module for synthesizing source-local triplet hubs."""

import dspy

from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHubDefinition,
    SourceTripletHubSummary,
    SourceTripletHubSummaryInput,
    SourceTripletHubSummaryMergeInput,
    SourceTripletHubSummarySynthesisInput,
    SourceTripletHubSynthesisInput,
)


class SourceTripletHubSignature(dspy.Signature):
    r"""Synthesize one source-local assertion from fixed or summarized evidence.

    Accept original evidence or ordered temporary summaries. Return a concise
    canonical assertion and standalone explanation grounded only in that
    evidence. Preserve directed relations and qualified, negative, conditional,
    quantified, and mathematical content. Final naming occurs only here. Never
    select membership, infer consequences, merge neighboring facts, or
    introduce unsupported facts.
    """

    request: (
        SourceTripletHubSynthesisInput | SourceTripletHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed source triplet evidence or ordered temporary summaries.'
    )
    definition: SourceTripletHubDefinition = dspy.OutputField(
        description='One canonical source-local triplet hub definition.'
    )


class SourceTripletHubModule(dspy.Module):
    """Synthesize one source-local triplet hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: SourceTripletHubSynthesisInput
        | SourceTripletHubSummarySynthesisInput,
    ) -> SourceTripletHubDefinition:
        """Synthesize a triplet hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourceTripletHubSynthesisInput
        | SourceTripletHubSummarySynthesisInput,
    ) -> SourceTripletHubDefinition:
        """Synthesize a triplet hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


class SourceTripletHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied triplet evidence.

    Preserve distinguishing content, directed relation, qualifications,
    negation, conditions, quantities, and mathematics; remove repetition
    without changing meaning. Keep role names and descriptions associated, and
    retain distinctions among role, source-fact, and triplet records. Do not
    choose a canonical hub name or change membership. Emit a concise,
    self-contained summary.
    """

    request: SourceTripletHubSummaryInput = dspy.InputField(
        description='Ordered triplet evidence to summarize.'
    )
    summary: SourceTripletHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary triplet evidence summary.'
    )


class SourceTripletHubSummaryModule(dspy.Module):
    """Summarize source triplet evidence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletHubSummaryInput
    ) -> SourceTripletHubSummary:
        """Summarize triplet evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceTripletHubSummaryInput
    ) -> SourceTripletHubSummary:
        """Summarize triplet evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceTripletHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered triplet summaries into one concise, self-contained summary.

    Consolidate repetition while preserving distinctions, qualifications,
    direction, negation, conditions, quantities, mathematics, and associations
    within the role records. Do not infer agreement or consequences. Do not
    create a hub definition or canonical name.
    """

    request: SourceTripletHubSummaryMergeInput = dspy.InputField(
        description='Ordered temporary triplet summaries.'
    )
    summary: SourceTripletHubSummary = dspy.OutputField(
        description='One consolidated temporary triplet summary.'
    )


class SourceTripletHubSummaryMergeModule(dspy.Module):
    """Merge source triplet summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletHubSummaryMergeInput
    ) -> SourceTripletHubSummary:
        """Merge triplet summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceTripletHubSummaryMergeInput
    ) -> SourceTripletHubSummary:
        """Merge triplet summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary
