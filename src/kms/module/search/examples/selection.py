"""Demonstrations for domain-aware final relevance selection."""

import dspy

from kms.core.model.content import MultimodalContent
from kms.core.model.search import (
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
                        material=MultimodalContent(
                            text='The derivative of x squared follows from the power rule.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='For f(x)=x^2, the difference quotient simplifies to 2x+h and tends to 2x.'
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
                        material=MultimodalContent(
                            text='Solve x^2 - 5x + 6 = 0.'
                        ),
                        is_exercise=True,
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='Solve x^2 - 5x + 6 = 0.'
                        ),
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
                        material=MultimodalContent(
                            text='Completing the square: move the constant, add the square of half the linear coefficient, then isolate the variable.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='A quadratic polynomial has degree two.'
                        ),
                    ),
                ],
            ),
            [0],
        ),
        (
            SearchSelectionInput(
                query='Compare claims about whether the sequence converges.',
                material_kind=SearchMaterialKind.PROPOSITION,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=MultimodalContent(
                            text='The sequence converges to zero.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='The sequence does not converge.'
                        ),
                    ),
                ],
            ),
            [0, 1],
        ),
        (
            SearchSelectionInput(
                query='Why does this argument establish the conclusion?',
                material_kind=SearchMaterialKind.ARGUMENT,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=MultimodalContent(
                            text='{"conclusion":"n is even","explanation":"The sum of two even integers is divisible by two.","premises":["a is even","b is even"]}'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='{"conclusion":"n is odd","explanation":"A product of odd integers is odd.","premises":["a is odd","b is odd"]}'
                        ),
                    ),
                ],
            ),
            [0],
        ),
        (
            SearchSelectionInput(
                query='Find an observation showing the learner confuses correlation with causation.',
                material_kind=SearchMaterialKind.LEARNER_OBSERVATION,
                candidates=[
                    SearchSelectionCandidate(
                        position=0,
                        material=MultimodalContent(
                            text='misconception | causality | Treats correlation as proof that one event caused the other.'
                        ),
                    ),
                    SearchSelectionCandidate(
                        position=1,
                        material=MultimodalContent(
                            text='demonstrated_understanding | causality | Explains that correlation alone does not establish causation.'
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
            content_images=[],
            selected_positions=positions,
        ).with_inputs('request', 'content_images')
        for request, positions in examples
    ]
