"""Demonstrations for domain-aware final relevance selection."""

import dspy

from kms.media.models import Material
from kms.source.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
    SearchSelectionInput,
)


def search_selection_examples() -> list[dspy.Example]:
    """Show useful promotion, rejection, and domain distinctions."""
    examples = [
        (
            SearchSelectionInput(
                query='Why does the derivative of x squared equal 2x?',
                material_kind=SearchMaterialKind.SOURCE_BLOCK,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=Material.from_text(
                            'The derivative of x squared follows from the power rule.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=Material.from_text(
                            'For f(x)=x^2, the difference quotient simplifies to 2x+h and tends to 2x.'
                        ),
                    ),
                ],
            ),
            [1, 0],
        ),
        (
            SearchSelectionInput(
                query='Find the exercise, not its worked solution.',
                material_kind=SearchMaterialKind.STATEMENT,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=Material.from_text('Solve x^2 - 5x + 6 = 0.'),
                        is_exercise=True,
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=Material.from_text('Solve x^2 - 5x + 6 = 0.'),
                        is_exercise=False,
                    ),
                ],
            ),
            [0],
        ),
        (
            SearchSelectionInput(
                query='How do I solve a quadratic by completing the square?',
                material_kind=SearchMaterialKind.PROCEDURE,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=Material.from_text(
                            'Completing the square: move the constant, add the square of half the linear coefficient, then isolate the variable.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=Material.from_text(
                            'A quadratic polynomial has degree two.'
                        ),
                    ),
                ],
            ),
            [0],
        ),
    ]
    return [
        dspy.Example(
            request=request,
            selected_positions=positions,
        ).with_inputs('request')
        for request, positions in examples
    ]
