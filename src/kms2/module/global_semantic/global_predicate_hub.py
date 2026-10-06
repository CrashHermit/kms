"""DSPy module for synthesizing global predicate hubs."""

import dspy

from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHubDefinition,
    GlobalPredicateHubSummary,
    GlobalPredicateHubSummaryInput,
    GlobalPredicateHubSummaryMergeInput,
    GlobalPredicateHubSummarySynthesisInput,
    GlobalPredicateHubSynthesisInput,
)


class GlobalPredicateHubSignature(dspy.Signature):
    r"""Synthesize one cross-source predicate relation from fixed evidence.

    Return a concise canonical predicate and description for supplied members
    or ordered partial summaries. Preserve predicate direction and meaning;
    never decide membership or use directed subject/object context. Final
    naming occurs only here.
    """

    request: (
        GlobalPredicateHubSynthesisInput
        | GlobalPredicateHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed predicate evidence; do not decide membership.'
    )
    definition: GlobalPredicateHubDefinition = dspy.OutputField(
        description='One canonical global predicate hub definition.'
    )


class GlobalPredicateHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied predicate evidence.

    Preserve each member's predicate/description association, direction,
    distinguishing content, conditions, negation, quantities, and
    qualifications. Remove repetition and emit a concise, self-contained
    temporary summary. Do not choose a canonical name or change membership.
    """

    request: GlobalPredicateHubSummaryInput = dspy.InputField(
        description='Ordered predicate evidence to summarize.'
    )
    summary: GlobalPredicateHubSummary = dspy.OutputField(
        description='One temporary predicate evidence summary.'
    )


class GlobalPredicateHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered predicate partial summaries into one temporary summary.

    Consolidate repetition while preserving distinctions, direction,
    qualifications, and member/field associations. Do not infer agreement or
    consequences, or produce a hub definition. Keep the merged summary concise.
    """

    request: GlobalPredicateHubSummaryMergeInput = dspy.InputField(
        description='Ordered predicate summaries to consolidate.'
    )
    summary: GlobalPredicateHubSummary = dspy.OutputField(
        description='One merged temporary predicate summary.'
    )


class GlobalPredicateHubSummaryModule(dspy.Module):
    """Generate one global predicate summary."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalPredicateHubSummaryInput
    ) -> GlobalPredicateHubSummary:
        """Summarize predicate evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalPredicateHubSummaryInput
    ) -> GlobalPredicateHubSummary:
        """Summarize predicate evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalPredicateHubSummaryMergeModule(dspy.Module):
    """Merge global predicate summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalPredicateHubSummaryMergeInput
    ) -> GlobalPredicateHubSummary:
        """Merge predicate summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalPredicateHubSummaryMergeInput
    ) -> GlobalPredicateHubSummary:
        """Merge predicate summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalPredicateHubModule(dspy.Module):
    """Synthesize one global predicate hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: GlobalPredicateHubSynthesisInput
        | GlobalPredicateHubSummarySynthesisInput,
    ) -> GlobalPredicateHubDefinition:
        """Synthesize a predicate hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: GlobalPredicateHubSynthesisInput
        | GlobalPredicateHubSummarySynthesisInput,
    ) -> GlobalPredicateHubDefinition:
        """Synthesize a predicate hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
