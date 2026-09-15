"""DSPy modules for discovering exercise components."""

import dspy

from kms2.core.model import BlockType, SourceBlockContext


def _block(
    content: str, block_type: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    """Build one live source-block demonstration context."""
    return SourceBlockContext(block_type=block_type, content=content)


def _set_demos(predictor: dspy.Module, demos: list[dspy.Example]) -> None:
    """Attach corrected live examples to the underlying predictor."""
    getattr(predictor, 'predictor', predictor).demos = demos


class ExerciseStartRouterSignature(dspy.Signature):
    r"""
    Classify only `target_block`. The before and after lists are context only.

    Return True only when the target begins one distinct individual exercise.
    Adjacent exercises are separate components even when they use the same
    topic, wording, or section. Return False for a shared instruction, a
    continuation or subpart of the preceding exercise, an attached table,
    image, equation, list, or explanatory paragraph, and ordinary prose.

    An exercise-numbered block that states a new task is a start even when it
    follows the same shared instruction and uses the same topic as the prior
    exercise. For example, target blocks beginning exercises 281 and 282 are
    both starts. Do not let topical similarity override the new exercise
    boundary.

    Return only a boolean. Do not return a reason, label, span, or text.
    Answer only the boolean True or False.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately before target_block; context only.'
    )
    target_block: SourceBlockContext = dspy.InputField(
        description='The only source block being classified as an individual exercise start.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField()
    is_exercise_start: bool = dspy.OutputField(
        description='True only when the designated block starts one individual exercise.'
    )


class ExerciseStartRouterModule(dspy.Module):
    """Route source blocks that begin individual exercise components."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    target_block=_block('280. Find the gradient of'),
                    context_after=[
                        _block(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                            BlockType.EQUATION,
                        ),
                        _block('Then, find the gradient at point $P(1, 2)$.'),
                    ],
                    target_block=_block('281. Find the gradient of'),
                    context_after=[
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        )
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_block(
                        'For the following exercises, find the gradient.'
                    ),
                    context_after=[_block('280. Find the gradient of')],
                    is_exercise_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[_block('280. Find the gradient of')],
                    target_block=_block(
                        '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                        BlockType.EQUATION,
                    ),
                    context_after=[
                        _block('Then, find the gradient at point $P(1, 2)$.')
                    ],
                    is_exercise_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        )
                    ],
                    target_block=_block(
                        '282. Find the gradient of $f(x, y, z)$ at $P$ and the '
                        'directional derivative in the direction of $\\mathbf{u}$:'
                    ),
                    context_after=[
                        _block(
                            '$$f(x, y, z) = \\ln(x^2 + 2y^2 + 3z^2), '
                            'P(2, 1, 4), \\mathbf{u} = '
                            '-\\frac{3}{13}\\mathbf{i} - \\frac{4}{13}\\mathbf{j} '
                            '- \\frac{12}{13}\\mathbf{k}.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the derivative of '
                            'the function at $P$ in the direction of $\\mathbf{u}$.'
                        )
                    ],
                    target_block=_block(
                        '286. $f(x, y) = -7x + 2y, P(2, -4), '
                        '\\mathbf{u} = 4\\mathbf{i} - 3\\mathbf{j}$'
                    ),
                    context_after=[
                        _block(
                            '287. $f(x, y) = \\ln(5x + 4y), P(3, 9), '
                            '\\mathbf{u} = 6\\mathbf{i} + 8\\mathbf{j}$'
                        )
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the derivative '
                            'of the function.'
                        )
                    ],
                    target_block=_block(
                        '294. $f(x, y) = x^2 + xy + y^2$ at point '
                        '$(-5, -4)$ in the direction the function increases most rapidly'
                    ),
                    context_after=[
                        _block(
                            '295. $f(x, y) = e^{xy}$ at point $(6, 7)$ in the '
                            'direction the function increases most rapidly'
                        )
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the maximum rate of '
                            'change of $f$ at the given point and the direction in '
                            'which it occurs.'
                        )
                    ],
                    target_block=_block('299. $f(x, y) = xe^{-y}, (1, 0)'),
                    context_after=[
                        _block('300. $f(x, y) = \\sqrt{x^2 + 2y}, (4, 10)')
                    ],
                    is_exercise_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
            ],
        )

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        target_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one exercise start synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_exercise_start

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        target_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one exercise start asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_exercise_start


class ExerciseBoundaryRouterSignature(dspy.Signature):
    r"""
    Classify only `candidate_block` relative to the individual exercise that
    begins at `start_block`. Use `context_after` only as context.

    Return False only when the candidate belongs to the anchored exercise,
    including its images, equations, tables, lists, explanatory blocks, or
    subparts. A leading exercise number followed by a period marks a new
    exercise unless the block is clearly a subpart or attached asset. Return
    True for such a new exercise even when the topic is similar. Return True
    for unrelated material.

    This is an exclusive boundary decision: the candidate at a boundary is not
    part of the anchored exercise. Do not describe it as an inclusive end and
    do not consume it. Return only a boolean. Answer only True or False.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one individual exercise.'
    )
    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately before candidate_block; context only.'
    )
    candidate_block: SourceBlockContext = dspy.InputField(
        description='The only source block being classified as a possible boundary.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately after candidate_block; context only.'
    )
    is_exercise_boundary: bool = dspy.OutputField(
        description='True when candidate_block does not belong to the anchored exercise.'
    )


class ExerciseBoundaryRouterModule(dspy.Module):
    """Route the exclusive boundary of one individual exercise."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
                dspy.Example(
                    start_block=_block('280. Find the gradient of'),
                    context_before=[
                        _block(
                            'For the following exercises, find the gradient.'
                        ),
                    ],
                    candidate_block=_block(
                        '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                        BlockType.EQUATION,
                    ),
                    context_after=[
                        _block('Then, find the gradient at point $P(1, 2).')
                    ],
                    is_exercise_boundary=False,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block('280. Find the gradient of'),
                    context_before=[
                        _block(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    candidate_block=_block(
                        'Then, find the gradient at point $P(1, 2).'
                    ),
                    context_after=[_block('281. Find the gradient of')],
                    is_exercise_boundary=False,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block('280. Find the gradient of'),
                    context_before=[
                        _block('Then, find the gradient at point $P(1, 2).')
                    ],
                    candidate_block=_block('281. Find the gradient of'),
                    context_after=[
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        )
                    ],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block('281. Find the gradient of'),
                    context_before=[
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        )
                    ],
                    candidate_block=_block(
                        '282. Find the gradient of $f(x, y, z)$ at $P$ and the '
                        'directional derivative in the direction of $\\mathbf{u}$:'
                    ),
                    context_after=[
                        _block(
                            '$$f(x, y, z) = \\ln(x^2 + 2y^2 + 3z^2), '
                            'P(2, 1, 4), \\mathbf{u} = '
                            '-\\frac{3}{13}\\mathbf{i} - \\frac{4}{13}\\mathbf{j} '
                            '- \\frac{12}{13}\\mathbf{k}.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '283. $f(x, y, z) = 4x^5 y^2 z^3, '
                        'P(2, -1, 1), \\mathbf{u} = '
                        '\\frac{1}{3}\\mathbf{i} + \\frac{2}{3}\\mathbf{j} '
                        '- \\frac{2}{3}\\mathbf{k}$'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        'For the following exercises, find the directional derivative '
                        'of the function at point $P$ in the direction of $Q$.'
                    ),
                    context_after=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '282. Find the gradient of $f(x, y, z)$ at $P$ and '
                        'the directional derivative in the direction of $\\mathbf{u}:'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '283. $f(x, y, z) = 4x^5 y^2 z^3, '
                        'P(2, -1, 1), \\mathbf{u} = '
                        '\\frac{1}{3}\\mathbf{i} + \\frac{2}{3}\\mathbf{j} '
                        '- \\frac{2}{3}\\mathbf{k}$'
                    ),
                    context_after=[
                        _block(
                            'For the following exercises, find the directional '
                            'derivative of the function at point $P$ in the direction of $Q$.'
                        )
                    ],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '285. $f(x, y, z) = \\frac{y}{x+z}, '
                        'P(2, 1, -1), Q(-1, 2, 0)$'
                    ),
                    context_after=[
                        _block(
                            'For the following exercises, find the derivative of '
                            'the function at $P$ in the direction of $\\mathbf{u}$.'
                        )
                    ],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '286. $f(x, y) = -7x + 2y, P(2, -4), '
                        '\\mathbf{u} = 4\\mathbf{i} - 3\\mathbf{j}$'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '287. $f(x, y) = \\ln(5x + 4y), P(3, 9), '
                        '\\mathbf{u} = 6\\mathbf{i} + 8\\mathbf{j}$'
                    ),
                    context_after=[],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '287. $f(x, y) = \\ln(5x + 4y), P(3, 9), '
                        '\\mathbf{u} = 6\\mathbf{i} + 8\\mathbf{j}$'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '288. [T] Use technology to sketch the level curve of '
                        '$f(x, y) = 4x - 2y + 3$ that passes through $P(1, 2)$ '
                        'and draw the gradient vector at $P$.'
                    ),
                    context_after=[],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '293. $f(x, y, z) = x\\sqrt{y^2 + z^2}, P(-2, -1, -1)$'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '294. $f(x, y) = x^2 + xy + y^2$ at point '
                        '$(-5, -4)$ in the direction the function increases most rapidly'
                    ),
                    context_after=[],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        '300. $f(x, y) = \\sqrt{x^2 + 2y}, (4, 10)'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '301. $f(x, y) = \\cos(3x + 2y), '
                        '\\left(\\frac{\\pi}{6}, -\\frac{\\pi}{8}\\right)'
                    ),
                    context_after=[],
                    is_exercise_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
            ],
        )

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        context_before: list[SourceBlockContext],
        candidate_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one exercise candidate synchronously."""
        prediction = self.predictor(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_exercise_boundary

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        context_before: list[SourceBlockContext],
        candidate_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one exercise candidate asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_exercise_boundary
