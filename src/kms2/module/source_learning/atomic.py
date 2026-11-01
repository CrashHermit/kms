"""DSPy module for content-grounded atomic source flashcards."""

import dspy

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardInput
from kms2.core.model.source_learning.flashcard import SourceFlashcardCandidate


class SourceAtomicFlashcardSignature(dspy.Signature):
    """Generate focused retrieval cards grounded in source material."""

    request: SourceAtomicFlashcardInput = dspy.InputField(
        description=(
            'Generate zero or more distinct question-and-answer cards from the '
            'source material and its exact evidence window. The target blocks '
            'are authoritative; before/after blocks may resolve references but '
            'cannot add independent claims. Test a useful definition, condition, '
            'distinction, explanation, or application without exceeding the '
            'supported material. Make every question and answer standalone. '
            'Preserve conditions, exceptions, negation, quantifiers, formulas, '
            'domains, defining objects, units, and notation. Do not omit a '
            'material qualification. Equivalent wording is acceptable only when '
            'it preserves the same meaning. Return [] when no useful nonduplicate '
            'card is warranted; do not manufacture an exhaustive inverse or a '
            'card merely to paraphrase the material. Use $...$ and $$...$$ for '
            'LaTeX only; never use backslash-parenthesis or backslash-bracket '
            'delimiters.'
        )
    )
    cards: list[SourceFlashcardCandidate] = dspy.OutputField(
        description='Zero or more content-only atomic flashcard candidates.'
    )


class SourceAtomicFlashcardModule(dspy.Module):
    """Generate atomic flashcards from one source-material evidence input."""

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
