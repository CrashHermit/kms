"""DSPy module for source-local statement descriptions."""

import dspy

from kms2.core.model import SourceStatementDescriptionInput


class SourceStatementDescriptionSignature(dspy.Signature):
    r"""Describe one complete source-local statement.

    Preserve the supported claim, definition, question, negation, quantifiers,
    constraints, and stated scope. Describe only target_blocks; neighboring
    blocks are reference context. Do not solve a problem or invent facts.
    """

    request: SourceStatementDescriptionInput = dspy.InputField(
        description=(
            'Describe only target_blocks. context_before and context_after are '
            'reference context, not independent evidence.'
        )
    )
    description: str = dspy.OutputField(
        description='One concise, source-grounded standalone statement description.'
    )


class SourceStatementDescriptionModule(dspy.Module):
    """Generate one source-local description for a statement."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: SourceStatementDescriptionInput) -> str:
        """Describe one source statement synchronously."""
        return self.predictor(request=request).description

    async def aforward(
        self, *, request: SourceStatementDescriptionInput
    ) -> str:
        """Describe one source statement asynchronously."""
        return (await self.predictor.acall(request=request)).description


__all__ = [
    'SourceStatementDescriptionModule',
    'SourceStatementDescriptionSignature',
]
