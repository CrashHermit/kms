"""DSPy module for synthesizing source-local procedure hubs."""

import dspy

from kms2.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHubDefinition,
    SourceProcedureHubSummary,
    SourceProcedureHubSummaryInput,
    SourceProcedureHubSummaryMergeInput,
    SourceProcedureHubSummarySynthesisInput,
    SourceProcedureHubSynthesisInput,
)


class SourceProcedureHubSignature(dspy.Signature):
    r"""Synthesize one source-local procedure from fixed or summarized evidence.

    Accept original procedure occurrences or ordered temporary summaries.
    Return a concise canonical name and source-grounded description. Preserve
    method order, steps, conditions, inputs, outputs, and termination. Final
    naming occurs only here. Never decide membership or invent facts.
    """

    request: (
        SourceProcedureHubSynthesisInput
        | SourceProcedureHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed procedure evidence or ordered temporary summaries.'
    )
    definition: SourceProcedureHubDefinition = dspy.OutputField(
        description='One canonical source-local procedure hub definition.'
    )


class SourceProcedureHubModule(dspy.Module):
    """Synthesize one source-local procedure hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: SourceProcedureHubSynthesisInput
        | SourceProcedureHubSummarySynthesisInput,
    ) -> SourceProcedureHubDefinition:
        """Synthesize a procedure hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: SourceProcedureHubSynthesisInput
        | SourceProcedureHubSummarySynthesisInput,
    ) -> SourceProcedureHubDefinition:
        """Synthesize a procedure hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition


class SourceProcedureHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied procedure evidence.

    Preserve distinguishing content, step order, conditions, inputs, outputs,
    and termination; remove repetition without changing meaning. Keep the full
    evidence for each member together. Do not choose a canonical name or change
    membership. Emit a concise, self-contained summary.
    """

    request: SourceProcedureHubSummaryInput = dspy.InputField(
        description='Ordered procedure evidence to summarize.'
    )
    summary: SourceProcedureHubSummary = dspy.OutputField(
        description='One concise, self-contained temporary procedure evidence summary.'
    )


class SourceProcedureHubSummaryModule(dspy.Module):
    """Summarize source procedure evidence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceProcedureHubSummaryInput
    ) -> SourceProcedureHubSummary:
        """Summarize procedure evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceProcedureHubSummaryInput
    ) -> SourceProcedureHubSummary:
        """Summarize procedure evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class SourceProcedureHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered procedure summaries into one concise, self-contained summary.

    Consolidate repetition while preserving distinctions, qualifications,
    method steps and order, conditions, inputs, outputs, termination, and their
    member associations. Do not infer agreement or consequences. Do not create
    a hub definition or canonical name.
    """

    request: SourceProcedureHubSummaryMergeInput = dspy.InputField(
        description='Ordered temporary procedure summaries.'
    )
    summary: SourceProcedureHubSummary = dspy.OutputField(
        description='One consolidated temporary procedure summary.'
    )


class SourceProcedureHubSummaryMergeModule(dspy.Module):
    """Merge source procedure summaries."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceProcedureHubSummaryMergeInput
    ) -> SourceProcedureHubSummary:
        """Merge procedure summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: SourceProcedureHubSummaryMergeInput
    ) -> SourceProcedureHubSummary:
        """Merge procedure summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary
