"""DSPy modules for synthesizing source-local entity hubs."""

import dspy

from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHubDefinition,
    SourceEntityHubSummary,
    SourceEntityHubSummaryInput,
    SourceEntityHubSummaryMergeInput,
    SourceEntityHubSummarySynthesisInput,
    SourceEntityHubSynthesisInput,
)


class SourceEntityHubSignature(dspy.Signature):
    r"""Synthesize one source-local concept from fixed entity evidence.

    Return a concise canonical name and source-grounded description for the
    supplied entity occurrences or ordered temporary summaries. Preserve
    mathematical notation separately from the concept it denotes. When the
    evidence supports the association, describe it with source-scoped wording
    such as "Velocity is represented by $v$ in this source." If different
    symbols denote the same supported concept, use the concept as the
    canonical name and mention the source-specific notation in the
    description. Never decide membership, add aliases from outside the
    members, infer equivalence from notation alone, or invent facts. Use
    `$...$` for inline LaTeX and `$$...$$` for display math; do not use
    alternate math delimiters.
    """

    request: (
        SourceEntityHubSynthesisInput | SourceEntityHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed entity evidence or reduced summaries; do not decide membership.'
    )
    definition: SourceEntityHubDefinition = dspy.OutputField(
        description='One canonical source-local entity hub definition.'
    )


class SourceEntityHubSummarySignature(dspy.Signature):
    r"""Summarize ordered entity evidence into a concise, self-contained summary.

    Summarize only the supplied evidence, preserving distinguishing content,
    conditions, negation, quantities, mathematics, direction, and associations
    between mathematical notation and its explicitly supported concepts while
    removing repetition. Keep each member's name and description associated.
    Do not choose a canonical hub name or change membership. Use `$...$` for
    inline LaTeX and `$$...$$` for display math; do not use alternate math
    delimiters.
    """

    request: SourceEntityHubSummaryInput = dspy.InputField(
        description='Ordered entity evidence to summarize.'
    )
    summary: SourceEntityHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary entity evidence summary.'
    )


class SourceEntityHubSummaryModule(dspy.Module):
    """Summarize source-local entity evidence in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityHubSummaryInput
    ) -> SourceEntityHubSummary:
        """Summarize entity evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceEntityHubSummaryInput
    ) -> SourceEntityHubSummary:
        """Summarize entity evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceEntityHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered temporary entity summaries without inventing agreement.

    Consolidate repetition while preserving distinctions, qualifications,
    member associations, and explicit associations between mathematical
    notation and concepts. Do not infer agreement or consequences. Produce one
    concise, self-contained temporary summary, not a hub definition.
    """

    request: SourceEntityHubSummaryMergeInput = dspy.InputField(
        description='Two or more ordered temporary entity summaries.'
    )
    summary: SourceEntityHubSummary = dspy.OutputField(
        description='One merged temporary entity evidence summary.'
    )


class SourceEntityHubSummaryMergeModule(dspy.Module):
    """Merge source-local entity summaries in one predictor call."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityHubSummaryMergeInput
    ) -> SourceEntityHubSummary:
        """Merge entity summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceEntityHubSummaryMergeInput
    ) -> SourceEntityHubSummary:
        """Merge entity summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceEntityHubModule(dspy.Module):
    """Synthesize one source-local entity hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: SourceEntityHubSynthesisInput
        | SourceEntityHubSummarySynthesisInput,
    ) -> SourceEntityHubDefinition:
        """Synthesize an entity hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourceEntityHubSynthesisInput
        | SourceEntityHubSummarySynthesisInput,
    ) -> SourceEntityHubDefinition:
        """Synthesize an entity hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
