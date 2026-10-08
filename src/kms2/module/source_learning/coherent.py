"""DSPy module for content-grounded coherent source flashcards."""

import dspy

from kms2.core.model.source_learning.coherent import (
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardInput,
)


class SourceCoherentFlashcardSignature(dspy.Signature):
    """Generate standalone cards for useful relationships among source ideas."""

    request: SourceCoherentFlashcardInput = dspy.InputField(
        description=(
            'Inspect the first-pass cards and every original triplet and source '
            'fact. Use the represented concepts and directed source relationships, '
            'not card wording alone, to identify meaningful connections. Generate '
            'zero or more additional focused cards only when a connection adds a '
            'new retrieval target beyond restating a parent card: for example a '
            'dependency, contrast, mechanism, explanation, application, or valid '
            'inference. Each card must test one meaningful target and stand alone. '
            'Never mention the source, evidence, first-pass cards, card indexes, '
            'or another card in the question or answer; card_indexes are metadata '
            'only. Use at least two parent cards and return the indexes actually '
            'used. Preserve relationship direction, assumptions, conditions, '
            'qualifiers, formulas, units, and notation. Generate only connections '
            'supported by the source or clearly defensible from it; do not invent '
            'facts or unsupported mathematical consequences. Return [] when no '
            'useful new card is warranted. Avoid duplicate retrieval targets. Use '
            '$...$ and $$...$$ for LaTeX only; never use backslash-parenthesis or '
            'backslash-bracket delimiters. This is the final generation pass; do '
            'not expect another pass.'
        )
    )
    cards: list[SourceCoherentFlashcardCandidate] = dspy.OutputField(
        description='Zero or more additional coherent flashcard candidates.'
    )


class SourceCoherentFlashcardModule(dspy.Module):
    """Generate coherent cards from retained atomic packets."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceCoherentFlashcardInput
    ) -> list[SourceCoherentFlashcardCandidate]:
        """Generate coherent cards synchronously."""
        return self.predictor(request=request).cards

    async def aforward(
        self, *, request: SourceCoherentFlashcardInput
    ) -> list[SourceCoherentFlashcardCandidate]:
        """Generate coherent cards asynchronously."""
        return (await self.predictor.acall(request=request)).cards


__all__ = ['SourceCoherentFlashcardModule', 'SourceCoherentFlashcardSignature']
