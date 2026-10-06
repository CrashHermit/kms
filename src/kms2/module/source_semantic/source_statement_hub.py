"""DSPy module for synthesizing source-local statement hubs."""

import dspy

from kms2.core.model.source_semantic.source_statement_hub import (
    SourceStatementHubDefinition,
    SourceStatementHubSummary,
    SourceStatementHubSummaryInput,
    SourceStatementHubSummaryMergeInput,
    SourceStatementHubSummarySynthesisInput,
    SourceStatementHubSynthesisInput,
)


class SourceStatementHubSummarySignature(dspy.Signature):
    r"""Summarize ordered source evidence for a statement hub.

    Preserve statement claim, theorem, question, and exercise qualifications.
    Emit a concise, self-contained summary of the supplied evidence.
    """

    request: SourceStatementHubSummaryInput = dspy.InputField(
        description='Ordered statement evidence to summarize.'
    )
    summary: SourceStatementHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary statement-hub evidence summary.'
    )


class SourceStatementHubSummaryModule(dspy.Module):
    """Generate one partial statement-hub summary."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceStatementHubSummaryInput
    ) -> SourceStatementHubSummary:
        """Generate a statement summary synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceStatementHubSummaryInput
    ) -> SourceStatementHubSummary:
        """Generate a statement summary asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceStatementHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered statement summaries without changing their meaning.

    Preserve qualifications and their associations without inferring agreement
    or consequences. Return one concise, self-contained temporary summary.
    """

    request: SourceStatementHubSummaryMergeInput = dspy.InputField(
        description='Ordered statement summaries to consolidate.'
    )
    summary: SourceStatementHubSummary = dspy.OutputField(
        description='One temporary merged statement-hub summary.'
    )


class SourceStatementHubSummaryMergeModule(dspy.Module):
    """Merge partial statement-hub summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceStatementHubSummaryMergeInput
    ) -> SourceStatementHubSummary:
        """Merge statement summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceStatementHubSummaryMergeInput
    ) -> SourceStatementHubSummary:
        """Merge statement summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceStatementHubSignature(dspy.Signature):
    r"""Synthesize one source-local statement from fixed evidence.

    Return a concise canonical name and source-grounded description. Preserve
    claim, theorem, question, and exercise qualifiers. The evidence is fixed;
    never decide membership or invent facts.
    """

    request: (
        SourceStatementHubSynthesisInput
        | SourceStatementHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed statement evidence or ordered partial summaries.'
    )
    definition: SourceStatementHubDefinition = dspy.OutputField(
        description='One canonical source-local statement hub definition.'
    )


class SourceStatementHubModule(dspy.Module):
    """Synthesize one source-local statement hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: SourceStatementHubSynthesisInput
        | SourceStatementHubSummarySynthesisInput,
    ) -> SourceStatementHubDefinition:
        """Synthesize a statement hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourceStatementHubSynthesisInput
        | SourceStatementHubSummarySynthesisInput,
    ) -> SourceStatementHubDefinition:
        """Synthesize a statement hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
