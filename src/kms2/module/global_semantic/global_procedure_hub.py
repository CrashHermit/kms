"""DSPy module for synthesizing global procedure hubs."""

import dspy

from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHubDefinition,
    GlobalProcedureHubSummary,
    GlobalProcedureHubSummaryInput,
    GlobalProcedureHubSummaryMergeInput,
    GlobalProcedureHubSummarySynthesisInput,
    GlobalProcedureHubSynthesisInput,
)


class GlobalProcedureHubSummarySignature(dspy.Signature):
    r"""Summarize only the ordered supplied evidence for a global procedure.

    Preserve each member's description association, distinguishing content,
    step order, conditions, inputs, outputs, and termination. Remove repetition
    and produce a concise, self-contained temporary summary. Do not choose a
    canonical hub name or change membership.
    """

    request: GlobalProcedureHubSummaryInput = dspy.InputField(
        description='Ordered fixed procedure evidence to summarize.'
    )
    summary: GlobalProcedureHubSummary = dspy.OutputField(
        description='One temporary procedure evidence summary.'
    )


class GlobalProcedureHubSummaryModule(dspy.Module):
    """Forward one typed global procedure summary request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalProcedureHubSummaryInput
    ) -> GlobalProcedureHubSummary:
        """Summarize procedure evidence synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalProcedureHubSummaryInput
    ) -> GlobalProcedureHubSummary:
        """Summarize procedure evidence asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalProcedureHubSummaryMergeSignature(dspy.Signature):
    r"""Merge ordered procedure partials without adding or inferring claims.

    Consolidate repetition while preserving distinctions, qualifications, and
    member/field associations. Produce one concise temporary summary, not a hub
    definition.
    """

    request: GlobalProcedureHubSummaryMergeInput = dspy.InputField(
        description='Ordered partial procedure summaries to consolidate.'
    )
    summary: GlobalProcedureHubSummary = dspy.OutputField(
        description='One temporary procedure evidence summary.'
    )


class GlobalProcedureHubSummaryMergeModule(dspy.Module):
    """Forward one typed global procedure summary-merge request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: GlobalProcedureHubSummaryMergeInput
    ) -> GlobalProcedureHubSummary:
        """Merge procedure summaries synchronously."""
        return self.predictor(request=request).summary

    async def aforward(
        self, *, request: GlobalProcedureHubSummaryMergeInput
    ) -> GlobalProcedureHubSummary:
        """Merge procedure summaries asynchronously."""
        return (await self.predictor.acall(request=request)).summary


class GlobalProcedureHubSignature(dspy.Signature):
    r"""Synthesize a global procedure from original evidence or ordered partials.

    Preserve step order, conditions, inputs, outputs, and termination. Evidence
    is fixed; never decide membership or invent claims. Do not include evidence
    field labels in the canonical definition.
    """

    request: (
        GlobalProcedureHubSynthesisInput
        | GlobalProcedureHubSummarySynthesisInput
    ) = dspy.InputField(
        description='Fixed procedure community evidence; do not decide membership.'
    )
    definition: GlobalProcedureHubDefinition = dspy.OutputField(
        description='One canonical global procedure hub definition.'
    )


class GlobalProcedureHubModule(dspy.Module):
    """Synthesize one global procedure hub definition."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: (
            GlobalProcedureHubSynthesisInput
            | GlobalProcedureHubSummarySynthesisInput
        ),
    ) -> GlobalProcedureHubDefinition:
        """Synthesize a procedure hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self,
        *,
        request: (
            GlobalProcedureHubSynthesisInput
            | GlobalProcedureHubSummarySynthesisInput
        ),
    ) -> GlobalProcedureHubDefinition:
        """Synthesize a procedure hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
