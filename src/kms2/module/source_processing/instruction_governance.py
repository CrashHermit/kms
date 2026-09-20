"""DSPy module for governing instruction-to-statement relationships."""

import dspy

from kms2.core.model import BlockType, SourceBlockContext


def _block(content: str) -> SourceBlockContext:
    """Build one live source-block demonstration context."""
    return SourceBlockContext(block_type=BlockType.PARAGRAPH, content=content)


def _set_demos(predictor: dspy.Module, demos: list[dspy.Example]) -> None:
    """Attach corrected live examples to the underlying predictor."""
    getattr(predictor, 'predictor', predictor).demos = demos


class InstructionGovernanceSignature(dspy.Signature):
    r"""
    TASK
    Decide whether this INSTRUCTION governs this STATEMENT.

    The statement is the complete semantic unit being evaluated. The context
    window contains the statement's blocks plus nearby document blocks that may
    clarify a lead-in, heading, qualifier, or section boundary.

    Return True if the instruction's directive applies to the statement.
    Return False if the statement is independent or belongs to another section.
    Return only a boolean. Answer only True or False.
    """

    instruction_blocks: list[SourceBlockContext] = dspy.InputField(
        description='Ordered complete instruction blocks. Image descriptions appear as text; no assets or bytes.'
    )
    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Ordered context blocks before statement_blocks; reference only.'
    )
    statement_blocks: list[SourceBlockContext] = dspy.InputField(
        description='Ordered complete statement blocks being governed.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Ordered context blocks after statement_blocks; reference only.'
    )
    governs: bool = dspy.OutputField(
        description='True if the instruction governs the statement.'
    )


class InstructionGovernanceModule(dspy.Module):
    """Judge instruction governance for complete exercise statements."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _set_demos(
            predictor,
            [
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block('280. Find the gradient of'),
                        _block('$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$'),
                        _block('Then, find the gradient at point $P(1, 2)$.'),
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '282. Find the gradient of $f(x, y, z)$ at $P and '
                            'the directional derivative in the direction of '
                            '$\\mathbf{u}:'
                        )
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the gradient.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    context_after=[],
                    governs=False,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the directional '
                            'derivative of the function at point $P$ in the '
                            'direction of $Q$.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the derivative '
                            'of the function at $P$ in the direction of '
                            '$\\mathbf{u}$.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '286. $f(x, y) = -7x + 2y, P(2, -4), '
                            '\\mathbf{u} = 4\\mathbf{i} - 3\\mathbf{j}$'
                        )
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the derivative '
                            'of the function at $P$ in the direction of '
                            '$\\mathbf{u}$.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '294. $f(x, y) = x^2 + xy + y^2$ at point '
                            '$(-5, -4)$ in the direction the function '
                            'increases most rapidly'
                        )
                    ],
                    context_after=[],
                    governs=False,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the maximum rate '
                            'of change of $f$ at the given point and the '
                            'direction in which it occurs.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block('299. $f(x, y) = x^2 + y^2$ at point $(1, 2)$')
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the derivative '
                            'of the function at $P$ in the direction of '
                            '$\\mathbf{u}$.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '288. [T] Use technology to sketch the level curve '
                            'of $f(x, y) = 4x - 2y + 3$ that passes through '
                            '$P(1, 2)$ and draw the gradient vector at $P$.'
                        )
                    ],
                    context_after=[],
                    governs=False,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block(
                            'For the following exercises, find the derivative '
                            'of the function at $P$ in the direction of '
                            '$\\mathbf{u}$.'
                        )
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '289. [T] Use technology to sketch the level curve '
                            'of $f(x, y) = x^2 + 4y^2$ that passes through '
                            '$P(-2, 0)$ and draw the gradient vector at $P$.'
                        )
                    ],
                    context_after=[],
                    governs=False,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
                dspy.Example(
                    instruction_blocks=[
                        _block('For the following exercises, find equations of')
                    ],
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '302. The level surface $$f(x, y, z) = 12$$ for '
                            '$$f(x, y, z) = 4x^2 - 2y^2 + z^2$$ at point '
                            '$(2, 2, 2)$.'
                        )
                    ],
                    context_after=[],
                    governs=True,
                ).with_inputs(
                    'instruction_blocks',
                    'context_before',
                    'statement_blocks',
                    'context_after',
                ),
            ],
        )

    def forward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        context_before: list[SourceBlockContext],
        statement_blocks: list[SourceBlockContext],
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Judge one instruction-statement relationship synchronously."""
        prediction = self.predictor(
            instruction_blocks=instruction_blocks,
            context_before=context_before,
            statement_blocks=statement_blocks,
            context_after=context_after,
        )
        return prediction.governs

    async def aforward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        context_before: list[SourceBlockContext],
        statement_blocks: list[SourceBlockContext],
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Judge one instruction-statement relationship asynchronously."""
        prediction = await self.predictor.acall(
            instruction_blocks=instruction_blocks,
            context_before=context_before,
            statement_blocks=statement_blocks,
            context_after=context_after,
        )
        return prediction.governs
