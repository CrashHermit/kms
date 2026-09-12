"""DSPy module for source-local entity occurrence enrichment."""

import dspy

from kms2.core.model import TermEnrichmentInput


class EntityEnrichmentSignature(dspy.Signature):
    r"""Describe one entity term in its supplied technical passage.

    Write a concise source-local gloss of what the noun phrase, object, or
    concept means in this passage. Resolve notation and local references, but
    do not create a general definition, merge synonyms, or invent facts.
    Describe only target_block; neighboring blocks are reference context.
    """

    request: TermEnrichmentInput = dspy.InputField(
        description=(
            'Describe only target_block. context_before and context_after are '
            'reference context, not independent evidence.'
        )
    )
    description: str = dspy.OutputField(
        description='One concise, source-grounded local description.'
    )


class EntityEnrichmentModule(dspy.Module):
    """Generate one source-local description for an entity occurrence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: TermEnrichmentInput) -> str:
        """Describe one entity occurrence synchronously."""
        return self.predictor(request=request).description

    async def aforward(self, *, request: TermEnrichmentInput) -> str:
        """Describe one entity occurrence asynchronously."""
        return (await self.predictor.acall(request=request)).description


__all__ = ['EntityEnrichmentModule', 'EntityEnrichmentSignature']
