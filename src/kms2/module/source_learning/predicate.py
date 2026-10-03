"""DSPy modules for source-predicate learning facts and cards."""

import dspy

from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactCandidate,
    SourcePredicateLearningFactInput,
)


class SourcePredicateLearningFactSignature(dspy.Signature):
    """Extract atomic learning facts focused on one directed relation."""

    request: SourcePredicateLearningFactInput = dspy.InputField(
        description=(
            'The single source_fact_text is the only evidence; hub_name identifies '
            'the focus and adds no claims. Return zero or more short, self-contained '
            'atomic facts preserving participants, qualifiers, negation, and notation. '
            'Return [] when this source fact contributes no fact for the focus. '
            'Do not add external context or support attribution.'
        )
    )
    facts: list[SourcePredicateLearningFactCandidate] = dspy.OutputField(
        description='Zero or more content-only predicate learning-fact candidates.'
    )


class SourcePredicateLearningFactModule(dspy.Module):
    """Extract typed learning facts for one predicate hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateLearningFactInput
    ) -> list[SourcePredicateLearningFactCandidate]:
        """Extract predicate learning facts synchronously."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: SourcePredicateLearningFactInput
    ) -> list[SourcePredicateLearningFactCandidate]:
        """Extract predicate learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).facts


class SourcePredicateFlashcardSignature(dspy.Signature):
    """Write one focused question and answer from one learning fact."""

    request: SourcePredicateFlashcardInput = dspy.InputField(
        description='One persisted predicate learning-fact text, without graph identity.'
    )
    result: SourcePredicateFlashcardResult = dspy.OutputField(
        description=(
            'Exactly one focused question and answer expressing the supplied fact '
            'without introducing claims. Do not rediscover, re-extract, select support, '
            'or add auxiliary context.'
        )
    )


class SourcePredicateFlashcardModule(dspy.Module):
    """Synthesize one card for one predicate learning fact."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateFlashcardInput
    ) -> SourcePredicateFlashcardResult:
        """Create a predicate card synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourcePredicateFlashcardInput
    ) -> SourcePredicateFlashcardResult:
        """Create a predicate card asynchronously."""
        return (await self.predictor.acall(request=request)).result
