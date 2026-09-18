"""DSPy module for source-faithful triplet decomposition."""

import dspy

from kms2.core.model import (
    TripletDecompositionCandidate,
    TripletDecompositionRequest,
)


class TripletDecompositionSignature(dspy.Signature):
    r"""Decompose one accepted source fact into explicit relations.

    Preserve source wording and mathematical notation. Use exact source
    phrases for subject and object, and a concise directed predicate. Classify
    persistent things, concepts, places, documents, and quantities as entity;
    classify explicitly named occurrences, actions, transitions, and state
    changes as event. Reject directives, questions, unsupported implications,
    unresolved pronouns, and facts without a defensible distinct subject,
    predicate, and object. Every accepted fact must produce at least one
    explicit triplet; do not return an empty list for an accepted fact.
    """

    fact_text: str = dspy.InputField(description='One accepted source fact.')
    triplets: list[TripletDecompositionCandidate] = dspy.OutputField(
        description='One or more explicit source-grounded triplets.'
    )


class TripletDecomposerModule(dspy.Module):
    """Run the subject-predicate-object decomposition pass."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: TripletDecompositionRequest
    ) -> list[TripletDecompositionCandidate]:
        """Decompose one source fact synchronously."""
        return self.predictor(fact_text=request.fact_text).triplets

    async def aforward(
        self, *, request: TripletDecompositionRequest
    ) -> list[TripletDecompositionCandidate]:
        """Decompose one source fact asynchronously."""
        return (
            await self.predictor.acall(fact_text=request.fact_text)
        ).triplets


__all__ = ['TripletDecompositionSignature', 'TripletDecomposerModule']
