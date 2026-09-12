"""DSPy module for source-local predicate occurrence enrichment."""

import dspy

from kms2.core.model import TermEnrichmentInput


class PredicateEnrichmentSignature(dspy.Signature):
    r"""Give one predicate term a compact, source-grounded meaning.

    Describe only the directed relation semantics. Preserve direction,
    negation, modality, causality, and temporal meaning when explicit. Do not
    restate endpoint names, formulas, values, or source details. Return a
    compact 2–12-word relation description, not a new fact or example.
    """

    request: TermEnrichmentInput = dspy.InputField(
        description=(
            'Describe only target_block predicate semantics. '
            'context_before and context_after are reference context.'
        )
    )
    description: str = dspy.OutputField(
        description='A compact 2–12-word semantic relation description.'
    )


class PredicateEnrichmentModule(dspy.Module):
    """Generate one source-local description for a predicate occurrence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: TermEnrichmentInput) -> str:
        """Describe one predicate occurrence synchronously."""
        return self.predictor(request=request).description

    async def aforward(self, *, request: TermEnrichmentInput) -> str:
        """Describe one predicate occurrence asynchronously."""
        return (await self.predictor.acall(request=request)).description


__all__ = ['PredicateEnrichmentModule', 'PredicateEnrichmentSignature']
