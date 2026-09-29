"""DSPy modules for source-event learning facts and cards."""

import dspy

from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFactInput,
    SourceEventLearningFactResult,
)


class SourceEventLearningFactSignature(dspy.Signature):
    """Extract independently testable facts about one event hub."""

    request: SourceEventLearningFactInput = dspy.InputField(
        description='Fixed source evidence for one event hub.'
    )
    result: SourceEventLearningFactResult = dspy.OutputField(
        description='Atomic source-faithful event facts with evidence UUIDs.'
    )


class SourceEventLearningFactModule(dspy.Module):
    """Extract typed learning facts for one event hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventLearningFactInput
    ) -> SourceEventLearningFactResult:
        """Extract event learning facts synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceEventLearningFactInput
    ) -> SourceEventLearningFactResult:
        """Extract event learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).result


class SourceEventFlashcardSignature(dspy.Signature):
    """Write one focused card from exactly one event learning fact."""

    request: SourceEventFlashcardInput = dspy.InputField(
        description='One persisted event learning fact and its fixed evidence.'
    )
    result: SourceEventFlashcardResult = dspy.OutputField(
        description='Exactly one grounded question and answer.'
    )


class SourceEventFlashcardModule(dspy.Module):
    """Synthesize one card for one event learning fact."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventFlashcardInput
    ) -> SourceEventFlashcardResult:
        """Create an event card synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceEventFlashcardInput
    ) -> SourceEventFlashcardResult:
        """Create an event card asynchronously."""
        return (await self.predictor.acall(request=request)).result
