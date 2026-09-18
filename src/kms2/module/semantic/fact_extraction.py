"""DSPy module for source-faithful fact extraction."""

import dspy

from kms2.core.model import AtomicFact, FactExtractionInput


class FactExtractionSignature(dspy.Signature):
    r"""Extract only explicit, durable facts asserted by the target block.

    Neighboring blocks resolve references only; they are never independent
    evidence. Preserve notation, participants, temporal expressions, negation,
    qualifiers, and causal wording. Split independent facts but keep an event
    with its participants and qualifiers. Every returned fact must support at
    least one explicit directed subject-predicate-object triplet. Return no
    facts for headings, metadata, questions, instructions, exercise requests,
    calculations, context-only claims, or claims without a defensible triplet.
    Return one short atomic fact per independent fact.
    """

    request: FactExtractionInput = dspy.InputField(
        description=(
            'target_block is the only source evidence. context_before and '
            'context_after may resolve references but cannot contribute facts.'
        )
    )
    facts: list[AtomicFact] = dspy.OutputField(
        description=(
            'Source-faithful atomic facts, each requiring at least one '
            'explicit triplet, or an empty list.'
        )
    )


class FactExtractorModule(dspy.Module):
    """Run the atomic source-fact extraction pass."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: FactExtractionInput) -> list[AtomicFact]:
        """Extract facts synchronously from one UUID-free local request."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: FactExtractionInput
    ) -> list[AtomicFact]:
        """Extract facts asynchronously from one UUID-free local request."""
        return (await self.predictor.acall(request=request)).facts


__all__ = ['FactExtractionSignature', 'FactExtractorModule']
