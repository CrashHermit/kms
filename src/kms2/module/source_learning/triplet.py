"""DSPy modules for source-triplet learning facts and cards."""

import dspy

from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFactCandidate,
    SourceTripletLearningFactInput,
)


class SourceTripletLearningFactSignature(dspy.Signature):
    """Extract atomic learning facts focused on one complete directed claim."""

    request: SourceTripletLearningFactInput = dspy.InputField(
        description=(
            'The single source_fact_text is the only evidence; hub_name identifies '
            'the focus and adds no claims. Return zero or more short, self-contained '
            'atomic facts preserving participants, qualifiers, negation, and notation. '
            'Return [] when this source fact contributes no fact for the focus. '
            'Do not add external context or support attribution.'
        )
    )
    facts: list[SourceTripletLearningFactCandidate] = dspy.OutputField(
        description='Zero or more content-only triplet learning-fact candidates.'
    )


class SourceTripletLearningFactModule(dspy.Module):
    """Extract typed learning facts for one triplet hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceTripletLearningFactInput
    ) -> list[SourceTripletLearningFactCandidate]:
        """Extract triplet learning facts synchronously."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: SourceTripletLearningFactInput
    ) -> list[SourceTripletLearningFactCandidate]:
        """Extract triplet learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).facts


class SourceTripletFlashcardSignature(dspy.Signature):
    """Write one focused question and answer from one learning fact."""

    request: SourceTripletFlashcardInput = dspy.InputField(
        description='One persisted triplet learning-fact text, without graph identity.'
    )
    result: SourceTripletFlashcardResult = dspy.OutputField(
        description=(
            'Exactly one focused question and answer expressing the supplied fact '
            'without introducing claims. Do not rediscover, re-extract, select support, '
            'or add auxiliary context.'
        )
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
