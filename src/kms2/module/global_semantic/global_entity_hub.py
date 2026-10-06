"""DSPy module for synthesizing global entity hubs."""

import dspy

from kms2.core.model.global_semantic.global_entity_hub import (
    GlobalEntityHubDefinition,
    GlobalEntityHubSummary,
    GlobalEntityHubSummaryInput,
    GlobalEntityHubSummaryMergeInput,
    GlobalEntityHubSummarySynthesisInput,
    GlobalEntityHubSynthesisInput,
)


class GlobalEntityHubSignature(dspy.Signature):
    r"""Synthesize a canonical global entity from fixed evidence.

    Accept either original source-hub evidence or ordered partial summaries.
    Define the entity's conceptual identity with a concise canonical name and
    description. The supplied evidence is fixed; never decide membership. Do
    not treat evidence field labels as domain claims.
    """

    request: (
        GlobalEntityHubSynthesisInput | GlobalEntityHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed entity evidence or ordered partial summaries; do not decide membership.'
    )
    definition: GlobalEntityHubDefinition = dspy.OutputField(
        description='One canonical global entity hub definition.'
    )


class GlobalEntityHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied global entity evidence.

    Preserve member and field associations, distinguishing content, conditions,
    negation, quantities, and mathematics. Remove repetition and emit a concise,
    self-contained partial summary. Do not choose a canonical hub name or
    change membership.
    """

    request: GlobalEntityHubSummaryInput = dspy.InputField(
        description='Ordered entity evidence records to summarize.'
    )
    summary: GlobalEntityHubSummary = dspy.OutputField(
        description='One temporary partial entity summary.'
    )


class GlobalEntityHubSummaryMergeSignature(dspy.Signature):
    r"""Consolidate ordered partial global entity summaries.

    Remove repetition while preserving distinctions, qualifications, and
    member/field associations. Do not infer agreement or consequences. Produce
    one concise temporary summary, not a hub definition.
    """

    request: GlobalEntityHubSummaryMergeInput = dspy.InputField(
        description='Ordered entity partial summaries to consolidate.'
    )
    summary: GlobalEntityHubSummary = dspy.OutputField(
        description='One consolidated temporary entity summary.'
    )


class GlobalEntityHubModule(dspy.Module):
    """Synthesize one global entity hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: GlobalEntityHubSynthesisInput
        | GlobalEntityHubSummarySynthesisInput,
    ) -> GlobalEntityHubDefinition:
        """Synthesize an entity hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: GlobalEntityHubSynthesisInput
        | GlobalEntityHubSummarySynthesisInput,
    ) -> GlobalEntityHubDefinition:
        """Synthesize an entity hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


class GlobalEntityHubSummaryModule(dspy.Module):
    """Summarize global entity evidence in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEntityHubSummaryInput
    ) -> GlobalEntityHubSummary:
        """Summarize entity evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalEntityHubSummaryInput
    ) -> GlobalEntityHubSummary:
        """Summarize entity evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalEntityHubSummaryMergeModule(dspy.Module):
    """Merge global entity summaries in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalEntityHubSummaryMergeInput
    ) -> GlobalEntityHubSummary:
        """Merge entity summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalEntityHubSummaryMergeInput
    ) -> GlobalEntityHubSummary:
        """Merge entity summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary
