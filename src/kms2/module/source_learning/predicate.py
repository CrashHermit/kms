"""DSPy modules for source-predicate learning facts and cards."""

import dspy

from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactResult,
)


class SourcePredicateLearningFactSignature(dspy.Signature):
    """Extract independently testable facts about one predicate hub."""

    request: SourcePredicateLearningFactInput = dspy.InputField(
        description='Fixed directed endpoint evidence for one predicate hub.'
    )
    result: SourcePredicateLearningFactResult = dspy.OutputField(
        description='Atomic source-faithful predicate facts with evidence UUIDs.'
    )


class SourcePredicateLearningFactModule(dspy.Module):
    """Extract typed learning facts for one predicate hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourcePredicateLearningFactInput
    ) -> SourcePredicateLearningFactResult:
        """Extract predicate learning facts synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourcePredicateLearningFactInput
    ) -> SourcePredicateLearningFactResult:
        """Extract predicate learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).result


class SourcePredicateFlashcardSignature(dspy.Signature):
    """Write one focused card from exactly one predicate learning fact."""

    request: SourcePredicateFlashcardInput = dspy.InputField(
        description='One persisted predicate learning fact and its fixed evidence.'
    )
    result: SourcePredicateFlashcardResult = dspy.OutputField(
        description='Exactly one grounded question and answer.'
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
