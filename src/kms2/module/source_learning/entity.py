"""DSPy modules for source-entity learning facts and cards."""

import dspy

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactResult,
)


class SourceEntityLearningFactSignature(dspy.Signature):
    """Extract independently testable facts about one entity hub."""

    request: SourceEntityLearningFactInput = dspy.InputField(
        description='Fixed source evidence for one entity hub.'
    )
    result: SourceEntityLearningFactResult = dspy.OutputField(
        description=(
            'Atomic source-faithful facts with source fact and triplet UUIDs; '
            'do not use context-only material.'
        )
    )


class SourceEntityLearningFactModule(dspy.Module):
    """Extract typed learning facts for one entity hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityLearningFactInput
    ) -> SourceEntityLearningFactResult:
        """Extract entity learning facts synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceEntityLearningFactInput
    ) -> SourceEntityLearningFactResult:
        """Extract entity learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).result


class SourceEntityFlashcardSignature(dspy.Signature):
    """Write one focused card from exactly one entity learning fact."""

    request: SourceEntityFlashcardInput = dspy.InputField(
        description='One persisted entity learning fact and its fixed evidence.'
    )
    result: SourceEntityFlashcardResult = dspy.OutputField(
        description='Exactly one grounded question and answer.'
    )


class SourceEntityFlashcardModule(dspy.Module):
    """Synthesize one card for one entity learning fact."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityFlashcardInput
    ) -> SourceEntityFlashcardResult:
        """Create an entity card synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceEntityFlashcardInput
    ) -> SourceEntityFlashcardResult:
        """Create an entity card asynchronously."""
        return (await self.predictor.acall(request=request)).result
