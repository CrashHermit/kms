"""DSPy module for content-grounded coherent source flashcards."""

import dspy

from kms2.core.model.source_learning.coherent import (
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardInput,
)


class SourceCoherentFlashcardSignature(dspy.Signature):
    """Generate useful cards from distinct source-material packets."""

    request: SourceCoherentFlashcardInput = dspy.InputField(
        description=(
            'Inspect the original source evidence and the selected atomic cards '
            'within one source-material group. Generate an additional standalone '
            'card only when combining distinct material facets yields a useful '
            'target that no selected parent already tests. Paraphrases or '
            'equivalent formulations are not distinct facets and are never a '
            'reason to manufacture another card. Do not infer unsupported '
            'premise/conclusion structure. Preserve assumptions, conditions, '
            'exceptions, negation, quantifiers, domains, formulas, and symbol '
            'roles. Use at least two selected parent cards and return exactly '
            'their one-based indexes. Do not refer to evidence, cards, indexes, '
            'or the source in the question or answer. Return [] when no '
            'genuinely new useful target exists. Avoid duplicate retrieval '
            'targets. Use $...$ and $$...$$ for LaTeX only; never use '
            'backslash-parenthesis or backslash-bracket delimiters.'
        )
    )
    cards: list[SourceCoherentFlashcardCandidate] = dspy.OutputField(
        description='Zero or more additional coherent flashcard candidates.'
    )


class SourceCoherentFlashcardModule(dspy.Module):
    """Generate coherent flashcards from retained source-material packets."""

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
