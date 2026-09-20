"""DSPy modules for discovering pedagogical source components."""

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


class PedagogicalStartRouterSignature(dspy.Signature):
    r"""
    Classify only `target_block` as the possible start of one pedagogical
    component. The before and after lists are context only.

    Return True when the target starts a labelled or numbered definition,
    theorem, proposition, lemma, corollary, axiom, law, model, rule,
    principle, worked example, exercise, problem, or prescribed procedure. A
    procedure lead-in and its numbered steps form one component.

    Return False for headings, ordinary narrative, shared exercise
    instructions, continuations, dangling OCR fragments, isolated answer
    choices, formatting artifacts, and material already owned by an earlier
    component. Do not merge distinct base numbers or attach unrelated nearby
    text.

    Return only a boolean. Answer only True or False.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Ordered source blocks immediately before target_block; context only.'
    )
    target_block: SourceBlockContext = dspy.InputField(
        description='The only source block being classified as a pedagogical component start.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField()
    is_pedagogical_start: bool = dspy.OutputField(
        description='True only when target_block starts one pedagogical component.'
    )


class PedagogicalStartRouterModule(dspy.Module):
    """Route source blocks that begin pedagogical components."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
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
                    is_pedagogical_start=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_block(
                        'For the following exercises, find the gradient.'
                    ),
                    context_after=[_block('280. Find the gradient of')],
                    is_pedagogical_start=False,
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
                    is_pedagogical_start=True,
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
                        _block('Then, find the gradient at point $P(1, 2).')
                    ],
                    is_pedagogical_start=False,
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
        """Classify one pedagogical start synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_pedagogical_start

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        target_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one pedagogical start asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.is_pedagogical_start


class PedagogicalBoundaryRouterSignature(dspy.Signature):
    r"""
    Classify only `candidate_block` relative to the pedagogical component that
    begins at `start_block`. Use `context_after` only as context.

    Return False only when the candidate continues the anchored definition,
    theorem, proposition, lemma, corollary, axiom, law, model, rule, principle,
    worked example, exercise, problem, or prescribed procedure. Return True
    when the candidate starts a distinct component or is unrelated material.

    This is an exclusive boundary decision: the candidate at a boundary is
    not part of the anchored component. Do not describe it as an inclusive end
    and do not consume it. Return only a boolean. Answer only True or False.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one pedagogical component.'
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
    is_pedagogical_boundary: bool = dspy.OutputField(
        description='True when candidate_block does not belong to the anchored component.'
    )


class PedagogicalBoundaryRouterModule(dspy.Module):
    """Route the exclusive boundary of one pedagogical component."""

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
                        )
                    ],
                    candidate_block=_block(
                        '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                        BlockType.EQUATION,
                    ),
                    context_after=[
                        _block('Then, find the gradient at point $P(1, 2).')
                    ],
                    is_pedagogical_boundary=False,
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
                    is_pedagogical_boundary=True,
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
                    is_pedagogical_boundary=True,
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
        """Classify one pedagogical candidate synchronously."""
        prediction = self.predictor(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_pedagogical_boundary

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        context_before: list[SourceBlockContext],
        candidate_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one pedagogical candidate asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block,
            context_before=context_before,
            candidate_block=candidate_block,
            context_after=context_after,
        )
        return prediction.is_pedagogical_boundary
