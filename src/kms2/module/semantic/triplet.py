"""DSPy modules for source-faithful fact and triplet extraction."""

import dspy

from kms2.core.model import (
    AtomicFact,
    FactExtractionInput,
    TripletCandidate,
    TripletDecompositionRequest,
)


class FactExtractionSignature(dspy.Signature):
    r"""Extract only explicit, durable facts asserted by the target block.

    Neighboring blocks resolve references only; they are never independent
    evidence. Preserve notation, participants, temporal expressions, negation,
    qualifiers, and causal wording. Split independent assertions but keep an
    event with its participants and qualifiers. Return no facts for headings,
    metadata, questions, instructions, exercise requests, calculations, or
    claims that appear only in context. Return one short atomic relation per
    independent assertion.
    """

    request: FactExtractionInput = dspy.InputField(
        description=(
            'The target_block is authoritative. context_before and '
            'context_after are reference context only.'
        )
    )
    facts: list[AtomicFact] = dspy.OutputField(
        description='Source-faithful atomic facts, or an empty list.'
    )


class TripletDecompositionSignature(dspy.Signature):
    r"""Decompose one accepted source fact into explicit relations.

    Preserve source wording and mathematical notation. Use exact source
    phrases for subject and object, and a concise directed predicate. Classify
    persistent things, concepts, places, documents, and quantities as entity;
    classify explicitly named occurrences, actions, transitions, and state
    changes as event. Reject directives, questions, unsupported implications,
    unresolved pronouns, and facts without a defensible distinct subject,
    predicate, and object. Return an empty list instead of guessing.
    """

    fact_text: str = dspy.InputField(description='One accepted source fact.')
    triplets: list[TripletCandidate] = dspy.OutputField(
        description='Explicit source-grounded triplets, or an empty list.'
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


class TripletDecomposerModule(dspy.Module):
    """Run the subject-predicate-object decomposition pass."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: TripletDecompositionRequest
    ) -> list[TripletCandidate]:
        """Decompose one source fact synchronously."""
        return self.predictor(fact_text=request.fact_text).triplets

    async def aforward(
        self, *, request: TripletDecompositionRequest
    ) -> list[TripletCandidate]:
        """Decompose one source fact asynchronously."""
        return (
            await self.predictor.acall(fact_text=request.fact_text)
        ).triplets


__all__ = [
    'FactExtractionSignature',
    'FactExtractorModule',
    'TripletDecompositionSignature',
    'TripletDecomposerModule',
]
