"""DSPy modules for discovering shared source instructions."""

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


class InstructionStartRouterSignature(dspy.Signature):
    r"""
    Classify only `target_block`. The before and after lists are context only.

    Return True only when the target block is an unnumbered directive
    introducing or governing multiple exercises. Return False for a numbered
    exercise, lettered fragment without its lead-in, ordinary prose, heading,
    answer, or continuation of an earlier instruction.

    Numbering takes precedence over imperative wording: a block such as
    `280. Find the gradient of ...` is an exercise, not an instruction,
    even when it begins with "Find", "Calculate", "Determine", or another
    command. Do not classify a numbered exercise as an instruction because a
    preceding block says "For the following exercises ...".

    Return only a boolean. Do not return a reason, label, span, or text.
    Answer only the boolean True or False.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately before target_block; context only.'
    )
    target_block: SourceBlockContext = dspy.InputField(
        description='The only source block being classified as an instruction start.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately after target_block; context only.'
    )
    is_instruction_start: bool = dspy.OutputField(
        description='True only when the designated block starts a shared exercise instruction.'
    )


class InstructionStartRouterModule(dspy.Module):
    """Route source blocks that begin shared instructions."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
                dspy.Example(
                    context_before=[],
                    target_block=_block(
                        'For the following exercises, find the gradient.'
                    ),
                    context_after=[_block('280. Find the gradient of')],
                    is_instruction_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
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
                    is_instruction_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_block(
                        '# 4.6 \\textbullet Directional Derivatives and the Gradient',
                        BlockType.HEADER,
                    ),
                    context_after=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    is_instruction_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[_block('283. $f(x, y, z) = 4x^5 y^2 z^3')],
                    target_block=_block(
                        'For the following exercises, find the directional derivative '
                        'of the function at point $P$ in the direction of $Q$.'
                    ),
                    context_after=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    is_instruction_start=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    target_block=_block('281. Find the gradient of'),
                    context_after=[
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        )
                    ],
                    is_instruction_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[_block('304. $f(x, y, z) = xyz = 6')],
                    target_block=_block(
                        '305. $f(x, y, z) = xe^y \\cos z - z = 1 '
                        'at point $(1, 0, 0)$'
                    ),
                    context_after=[
                        _block(
                            '306. The temperature $T$ in a metal sphere is '
                            'inversely proportional to the distance from the center.'
                        )
                    ],
                    is_instruction_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    target_block=_block(
                        '282. Find the gradient of $f(x, y, z) at $P and the '
                        'directional derivative in the direction of $\\mathbf{u}:'
                    ),
                    context_after=[
                        _block(
                            '$$f(x, y, z) = x^2 + y^2 + z^2.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_instruction_start=False,
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
        """Classify one instruction start synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_instruction_start

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        target_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one instruction start asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_instruction_start


class InstructionBoundaryRouterSignature(dspy.Signature):
    r"""
    Classify only `candidate_block` relative to the shared instruction that
    begins at `start_block`. The before and after lists are context only.

    Return False only when the candidate is a grammatical continuation of the
    anchored shared directive. Return True when the candidate starts an
    individual exercise, a new shared instruction, or unrelated content.

    Numbering takes precedence over imperative wording: a candidate such as
    `280. Find the gradient of ...` is an individual exercise, not a
    continuation of the shared directive, even when the shared directive says
    "For the following exercises, find ...". Return True for that candidate.
    A candidate containing a specific exercise number, expression, quantity,
    point, equation, question, or requested operation is normally an exercise,
    not instruction continuation. Do not use numbering as the sole criterion:
    apply the same boundary to unlabeled tasks.

    The candidate at a boundary is not part of the anchored instruction. Return
    only a boolean. Answer only True or False.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one shared instruction.'
    )
    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Source blocks immediately before candidate_block; context only.'
    )
    candidate_block: SourceBlockContext = dspy.InputField(
        description='The only source block being classified as a possible boundary.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Source blocks immediately after candidate_block; context only.'
    )
    is_instruction_boundary: bool = dspy.OutputField(
        description='True when candidate_block does not belong to the anchored instruction.'
    )


class InstructionBoundaryRouterModule(dspy.Module):
    """Route the exclusive boundary of one shared instruction."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find the gradient.'
                    ),
                    context_before=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    candidate_block=_block('280. Find the gradient of'),
                    context_after=[
                        _block(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_instruction_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find equations of'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        'a. the tangent plane and b. the normal line to the'
                    ),
                    context_after=[
                        _block(
                            'b. the normal line to the given surface at the given point.'
                        )
                    ],
                    is_instruction_boundary=False,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find the directional derivative '
                        'of the function at point $P$ in the direction of $Q$.'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$'
                    ),
                    context_after=[],
                    is_instruction_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find the derivative of '
                        'the function at $P$ in the direction of $\\mathbf{u}$.'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '286. $f(x, y) = -7x + 2y, P(2, -4), '
                        '\\mathbf{u} = 4\\mathbf{i} - 3\\mathbf{j}$'
                    ),
                    context_after=[
                        _block(
                            '287. $f(x, y) = \\ln(5x + 4y), P(3, 9), '
                            '\\mathbf{u} = 6\\mathbf{i} + 8\\mathbf{j}$'
                        )
                    ],
                    is_instruction_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find the derivative '
                        'of the function.'
                    ),
                    context_before=[],
                    candidate_block=_block(
                        '294. $f(x, y) = x^2 + xy + y^2$ at point '
                        '$(-5, -4)$ in the direction the function increases most rapidly'
                    ),
                    context_after=[],
                    is_instruction_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find the gradient.'
                    ),
                    context_before=[
                        _block('281. Find the gradient of'),
                        _block(
                            'Find the gradient of $f(x, y, z) = xy + yz + xz '
                            'at point $P(1, 2, 3)$.',
                            BlockType.EQUATION,
                        ),
                    ],
                    candidate_block=_block(
                        '282. Find the gradient of $f(x, y, z) at $P and the '
                        'directional derivative in the direction of $\\mathbf{u}:'
                    ),
                    context_after=[
                        _block(
                            '$$f(x, y, z) = x^2 + y^2 + z^2.$$',
                            BlockType.EQUATION,
                        )
                    ],
                    is_instruction_boundary=True,
                ).with_inputs(
                    'start_block',
                    'context_before',
                    'candidate_block',
                    'context_after',
                ),
                dspy.Example(
                    start_block=_block(
                        'For the following exercises, find equations of'
                    ),
                    context_before=[
                        _block(
                            'a. the tangent plane and\nb. the normal line to the'
                        )
                    ],
                    candidate_block=_block(
                        'b. the normal line to the given surface at the given point.'
                    ),
                    context_after=[_block('386', BlockType.HEADER)],
                    is_instruction_boundary=False,
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
        """Classify one instruction candidate synchronously."""
        prediction = self.predictor(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_instruction_boundary

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        context_before: list[SourceBlockContext],
        candidate_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one instruction candidate asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_instruction_boundary
