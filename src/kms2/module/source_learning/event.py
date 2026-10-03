"""DSPy modules for source-event learning facts and cards."""

import dspy

from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFactCandidate,
    SourceEventLearningFactInput,
)


class SourceEventLearningFactSignature(dspy.Signature):
    """Extract atomic learning facts focused on one occurrence or process."""

    request: SourceEventLearningFactInput = dspy.InputField(
        description=(
            'The single source_fact_text is the only evidence; hub_name identifies '
            'the focus and adds no claims. Return zero or more short, self-contained '
            'atomic facts preserving participants, qualifiers, negation, and notation. '
            'Return [] when this source fact contributes no fact for the focus. '
            'Do not add external context or support attribution.'
        )
    )
    facts: list[SourceEventLearningFactCandidate] = dspy.OutputField(
        description='Zero or more content-only event learning-fact candidates.'
    )


class SourceEventLearningFactModule(dspy.Module):
    """Extract typed learning facts for one event hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEventLearningFactInput
    ) -> list[SourceEventLearningFactCandidate]:
        """Extract event learning facts synchronously."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: SourceEventLearningFactInput
    ) -> list[SourceEventLearningFactCandidate]:
        """Extract event learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).facts


class SourceEventFlashcardSignature(dspy.Signature):
    """Write one focused question and answer from one learning fact."""

    request: SourceEventFlashcardInput = dspy.InputField(
        description='One persisted event learning-fact text, without graph identity.'
    )
    result: SourceEventFlashcardResult = dspy.OutputField(
        description=(
            'Exactly one focused question and answer expressing the supplied fact '
            'without introducing claims. Do not rediscover, re-extract, select support, '
            'or add auxiliary context.'
        )
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
