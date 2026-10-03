"""DSPy modules for source-entity learning facts and cards."""

import dspy

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFactCandidate,
    SourceEntityLearningFactInput,
)


class SourceEntityLearningFactSignature(dspy.Signature):
    """Extract atomic learning facts focused on one entity or concept."""

    request: SourceEntityLearningFactInput = dspy.InputField(
        description=(
            'The single source_fact_text is the only evidence; hub_name identifies '
            'the focus and adds no claims. Return zero or more short, self-contained '
            'atomic facts preserving participants, qualifiers, negation, and notation. '
            'Return [] when this source fact contributes no fact for the focus. '
            'Do not add external context or support attribution.'
        )
    )
    facts: list[SourceEntityLearningFactCandidate] = dspy.OutputField(
        description='Zero or more content-only entity learning-fact candidates.'
    )


class SourceEntityLearningFactModule(dspy.Module):
    """Extract typed learning facts for one entity hub."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceEntityLearningFactInput
    ) -> list[SourceEntityLearningFactCandidate]:
        """Extract entity learning facts synchronously."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: SourceEntityLearningFactInput
    ) -> list[SourceEntityLearningFactCandidate]:
        """Extract entity learning facts asynchronously."""
        return (await self.predictor.acall(request=request)).facts


class SourceEntityFlashcardSignature(dspy.Signature):
    """Write one focused question and answer from one learning fact."""

    request: SourceEntityFlashcardInput = dspy.InputField(
        description='One persisted entity learning-fact text, without graph identity.'
    )
    result: SourceEntityFlashcardResult = dspy.OutputField(
        description=(
            'Exactly one focused question and answer expressing the supplied fact '
            'without introducing claims. Do not rediscover, re-extract, select support, '
            'or add auxiliary context.'
        )
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
