"""DSPy module for synthesizing global statement hubs."""

import dspy

from kms2.core.model.global_semantic.global_statement_hub import (
    GlobalStatementHubDefinition,
    GlobalStatementHubSummary,
    GlobalStatementHubSummaryInput,
    GlobalStatementHubSummaryMergeInput,
    GlobalStatementHubSummarySynthesisInput,
    GlobalStatementHubSynthesisInput,
)


class GlobalStatementHubSignature(dspy.Signature):
    r"""Synthesize one cross-source statement from fixed evidence.

    Return a concise canonical name and description for supplied members or
    ordered partial summaries. Preserve whether each statement is a claim,
    fact, theorem, explanation, question, or exercise, including its
    qualifications. The members are fixed evidence; never decide membership or
    invent evidence. Final naming occurs only here.
    """

    request: (
        GlobalStatementHubSynthesisInput
        | GlobalStatementHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed statement evidence; preserve qualifiers and do not decide membership.'
    )
    definition: GlobalStatementHubDefinition = dspy.OutputField(
        description='One canonical global statement hub definition.'
    )


class GlobalStatementHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied statement evidence.

    Preserve each member's description association, distinguishing content,
    conditions, negation, quantities, mathematics, and statement qualifications.
    Remove repetition and emit a concise, self-contained temporary summary. Do
    not choose a canonical hub name or change membership.
    """

    request: GlobalStatementHubSummaryInput = dspy.InputField(
        description='Ordered statement evidence to summarize.'
    )
    summary: GlobalStatementHubSummary = dspy.OutputField(
        description='One temporary statement evidence summary.'
    )


class GlobalStatementHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered statement partial summaries into one temporary summary.

    Consolidate repetition while preserving distinctions, qualifications, and
    member/field associations without inferring agreement or consequences.
    Preserve statement types and qualifiers; do not produce a hub definition.
    """

    request: GlobalStatementHubSummaryMergeInput = dspy.InputField(
        description='Ordered statement summaries to consolidate.'
    )
    summary: GlobalStatementHubSummary = dspy.OutputField(
        description='One merged temporary statement summary.'
    )


class GlobalStatementHubSummaryModule(dspy.Module):
    """Generate one global statement summary."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalStatementHubSummaryInput
    ) -> GlobalStatementHubSummary:
        """Summarize statement evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalStatementHubSummaryInput
    ) -> GlobalStatementHubSummary:
        """Summarize statement evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalStatementHubSummaryMergeModule(dspy.Module):
    """Merge global statement summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalStatementHubSummaryMergeInput
    ) -> GlobalStatementHubSummary:
        """Merge statement summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalStatementHubSummaryMergeInput
    ) -> GlobalStatementHubSummary:
        """Merge statement summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalStatementHubModule(dspy.Module):
    """Synthesize one global statement hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: GlobalStatementHubSynthesisInput
        | GlobalStatementHubSummarySynthesisInput,
    ) -> GlobalStatementHubDefinition:
        """Synthesize a statement hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: GlobalStatementHubSynthesisInput
        | GlobalStatementHubSummarySynthesisInput,
    ) -> GlobalStatementHubDefinition:
        """Synthesize a statement hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
