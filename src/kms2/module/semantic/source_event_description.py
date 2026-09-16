"""DSPy module for source-local event occurrence descriptions."""

import dspy

from kms2.core.model import SourceEventDescriptionInput


class SourceEventDescriptionSignature(dspy.Signature):
    r"""Describe one event term as the occurrence supported by the passage.

    Preserve supported participants and roles, temporal, spatial, duration,
    ordering, polarity, causal, and state-transition qualifiers. Describe the
    event itself, not merely its result, consequence, motivation, or nearby
    explanatory clause. Do not infer unsupported facts.
    """

    request: SourceEventDescriptionInput = dspy.InputField(
        description=(
            'Describe only target_block as an event occurrence. '
            'context_before and context_after are reference context.'
        )
    )
    description: str = dspy.OutputField(
        description='One concise, source-grounded event description.'
    )


class SourceEventDescriptionModule(dspy.Module):
    """Generate one source-local description for an event occurrence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: SourceEventDescriptionInput) -> str:
        """Describe one event occurrence synchronously."""
        return self.predictor(request=request).description

    async def aforward(self, *, request: SourceEventDescriptionInput) -> str:
        """Describe one event occurrence asynchronously."""
        return (await self.predictor.acall(request=request)).description


__all__ = ['SourceEventDescriptionModule', 'SourceEventDescriptionSignature']
