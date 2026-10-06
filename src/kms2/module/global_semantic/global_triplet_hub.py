"""DSPy module for synthesizing global triplet hubs."""

import dspy

from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHubDefinition,
    GlobalTripletHubSummary,
    GlobalTripletHubSummaryInput,
    GlobalTripletHubSummaryMergeInput,
    GlobalTripletHubSummarySynthesisInput,
    GlobalTripletHubSynthesisInput,
)


class GlobalTripletHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied global triplet evidence.

    Field and role labels identify evidence context, not new claims. Preserve
    directed subject/predicate/object roles and distinguishing qualified,
    negative, conditional, and quantified content with its role/member
    associations. Remove repetition and produce a concise, self-contained
    temporary summary. Do not name a canonical hub or change membership.
    """

    request: GlobalTripletHubSummaryInput = dspy.InputField(
        description='Ordered fixed global triplet evidence to summarize.'
    )
    summary: GlobalTripletHubSummary = dspy.OutputField(
        description='One temporary global triplet evidence summary.'
    )


class GlobalTripletHubSummaryModule(dspy.Module):
    """Forward one typed global triplet summary request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalTripletHubSummaryInput
    ) -> GlobalTripletHubSummary:
        """Summarize global triplet evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalTripletHubSummaryInput
    ) -> GlobalTripletHubSummary:
        """Summarize global triplet evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalTripletHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered global triplet partials without adding claims.

    Consolidate repetition while preserving directed roles, distinctions,
    qualifications, and member/role associations without inferring agreement or
    consequences. Produce a concise temporary summary, not a hub definition.
    """

    request: GlobalTripletHubSummaryMergeInput = dspy.InputField(
        description='Ordered partial triplet summaries to consolidate.'
    )
    summary: GlobalTripletHubSummary = dspy.OutputField(
        description='One temporary global triplet evidence summary.'
    )


class GlobalTripletHubSummaryMergeModule(dspy.Module):
    """Forward one typed global triplet summary-merge request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalTripletHubSummaryMergeInput
    ) -> GlobalTripletHubSummary:
        """Merge global triplet summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalTripletHubSummaryMergeInput
    ) -> GlobalTripletHubSummary:
        """Merge global triplet summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalTripletHubSignature(dspy.Signature):
    r"""Synthesize a global relation from original evidence or ordered partials.

    Produce a canonical relation and standalone explanation grounded only in
    supplied roles and source-triplet-hub evidence. Preserve directed subject,
    predicate, and object roles and qualified, negative, conditional, and
    quantified claims. Do not infer consequences or add unsupported facts.
    """

    request: (
        GlobalTripletHubSynthesisInput | GlobalTripletHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed global triplet evidence; do not decide membership.'
    )
    definition: GlobalTripletHubDefinition = dspy.OutputField(
        description='One canonical global triplet hub definition.'
    )


class GlobalTripletHubModule(dspy.Module):
    """Synthesize one global triplet hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: (
            GlobalTripletHubSynthesisInput
            | GlobalTripletHubSummarySynthesisInput
        ),
    ) -> GlobalTripletHubDefinition:
        """Synthesize a global triplet hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: (
            GlobalTripletHubSynthesisInput
            | GlobalTripletHubSummarySynthesisInput
        ),
    ) -> GlobalTripletHubDefinition:
        """Synthesize a global triplet hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
