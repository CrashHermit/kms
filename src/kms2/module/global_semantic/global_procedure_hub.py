"""DSPy module for synthesizing global procedure hubs."""

import dspy

from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHubDefinition,
    GlobalProcedureHubSynthesisInput,
)


class GlobalProcedureHubSignature(dspy.Signature):
    r"""Synthesize one cross-source procedure from fixed evidence.

    Return a concise canonical name and description for the supplied source-hub
    members. Preserve ordered steps, conditions, inputs, outputs, and
    termination. The members are fixed evidence; never decide membership or
    invent evidence.
    """

    request: GlobalProcedureHubSynthesisInput = dspy.InputField(
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
        self, *, request: GlobalProcedureHubSynthesisInput
    ) -> GlobalProcedureHubDefinition:
        """Synthesize a procedure hub synchronously."""
        return self.predictor(request=request).definition

    async def aforward(
        self, *, request: GlobalProcedureHubSynthesisInput
    ) -> GlobalProcedureHubDefinition:
        """Synthesize a procedure hub asynchronously."""
        return (await self.predictor.acall(request=request)).definition
