"""DSPy module for content-grounded atomic source flashcards."""

import dspy

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardInput
from kms2.core.model.source_learning.flashcard import SourceFlashcardCandidate


class SourceAtomicFlashcardSignature(dspy.Signature):
    """Generate focused retrieval cards from one exact source claim."""

    request: SourceAtomicFlashcardInput = dspy.InputField(
        description=(
            'Generate zero or more distinct question-and-answer cards from the '
            'supplied directed relationship. Each card must test one meaningful '
            'retrieval target: a definition, condition, distinction, explanation, '
            'application, calculation, or other source-supported objective. A card '
            'may test a relationship; atomic and coherent describe generation '
            'stages, not mutually exclusive card types. The exact triplet and '
            'source fact are authoritative evidence. Hub descriptions explain '
            'concepts and notation but do not authorize unrelated claims. Make '
            'each question and answer standalone: never mention the source, '
            'evidence packet, or another card. Use the smallest complete answer. '
            'Preserve direction, negation, conditions, quantification, formulas, '
            'units, and source-supported symbol meanings. Accept equivalent '
            'wording when it expresses the same intended meaning. Return [] when '
            'no useful card is warranted. Do not force a card, reverse the '
            'relation exhaustively, or emit duplicate retrieval targets. Use '
            '$...$ and $$...$$ for LaTeX only; never use backslash-parenthesis or '
            'backslash-bracket delimiters.'
        )
    )
    cards: list[SourceFlashcardCandidate] = dspy.OutputField(
        description='Zero or more content-only atomic flashcard candidates.'
    )


class SourceAtomicFlashcardModule(dspy.Module):
    """Generate atomic flashcards from one exact triplet/fact input."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceAtomicFlashcardInput
    ) -> list[SourceFlashcardCandidate]:
        """Generate atomic cards synchronously."""
        return self.predictor(request=request).cards

    async def aforward(
        self, *, request: SourceAtomicFlashcardInput
    ) -> list[SourceFlashcardCandidate]:
        """Generate atomic cards asynchronously."""
        return (await self.predictor.acall(request=request)).cards


__all__ = ['SourceAtomicFlashcardModule', 'SourceAtomicFlashcardSignature']
