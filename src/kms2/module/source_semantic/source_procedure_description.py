"""DSPy module for source-local procedure descriptions."""

import dspy

from kms2.core.model import SourceProcedureDescriptionInput


class SourceProcedureDescriptionSignature(dspy.Signature):
    r"""Describe one ordered source-local procedure.

    Preserve explicit steps, ordering, conditions, inputs, outputs, and
    termination criteria. Describe only target_blocks; neighboring blocks are
    reference context. Do not infer omitted steps or unsupported facts.
    """

    request: SourceProcedureDescriptionInput = dspy.InputField(
        description=(
            'Describe only target_blocks. context_before and context_after are '
            'reference context, not independent evidence.'
        )
    )
    description: str = dspy.OutputField(
        description='One concise, source-grounded ordered procedure description.'
    )


class SourceProcedureDescriptionModule(dspy.Module):
    """Generate one source-local description for a procedure."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: SourceProcedureDescriptionInput) -> str:
        """Describe one source procedure synchronously."""
        return self.predictor(request=request).description

    async def aforward(
        self, *, request: SourceProcedureDescriptionInput
    ) -> str:
        """Describe one source procedure asynchronously."""
        return (await self.predictor.acall(request=request)).description


__all__ = [
    'SourceProcedureDescriptionModule',
    'SourceProcedureDescriptionSignature',
]
