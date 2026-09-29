"""DSPy modules for source-triplet learning facts and cards."""

import dspy

from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactResult,
)


class SourceTripletLearningFactSignature(dspy.Signature):
    """Extract independently testable facts about one triplet hub."""

    request: SourceTripletLearningFactInput = dspy.InputField(
        description='Fixed directed triplet evidence for one triplet hub.'
    )
    result: SourceTripletLearningFactResult = dspy.OutputField(
        description='Atomic source-faithful triplet facts with evidence UUIDs.'
    )


class SourceTripletLearningFactModule(dspy.Module):
    """Extract typed learning facts for one triplet hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletLearningFactInput
    ) -> SourceTripletLearningFactResult:
        """Extract triplet learning facts synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceTripletLearningFactInput
    ) -> SourceTripletLearningFactResult:
        """Extract triplet learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).result


class SourceTripletFlashcardSignature(dspy.Signature):
    """Write one focused card from exactly one triplet learning fact."""

    request: SourceTripletFlashcardInput = dspy.InputField(
        description='One persisted triplet learning fact and its fixed evidence.'
    )
    result: SourceTripletFlashcardResult = dspy.OutputField(
        description='Exactly one grounded question and answer.'
    )


class SourceTripletFlashcardModule(dspy.Module):
    """Synthesize one card for one triplet learning fact."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletFlashcardInput
    ) -> SourceTripletFlashcardResult:
        """Create a triplet card synchronously."""
        return self.predictor(request=request).result

    async def aforward(
        self, *, request: SourceTripletFlashcardInput
    ) -> SourceTripletFlashcardResult:
        """Create a triplet card asynchronously."""
        return (await self.predictor.acall(request=request)).result
