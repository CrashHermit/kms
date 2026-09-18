"""DSPy module for routing packed source exercises."""

import dspy

from kms2.core.model import BlockType, SourceBlockContext


def _context(content: str) -> SourceBlockContext:
    """Build one text-only splitter demonstration block."""
    return SourceBlockContext(block_type=BlockType.PARAGRAPH, content=content)


def _set_demos(predictor: dspy.Module, demos: list[dspy.Example]) -> None:
    """Attach demos to the underlying DSPy predictor when wrapped."""
    getattr(predictor, 'predictor', predictor).demos = demos


class ExerciseStripRouterSignature(dspy.Signature):
    r"""
    Classify only `target_block`. The before and after lists are context only.

    Return True only when the target visibly packs TWO OR MORE distinct
    numbered exercises or problems that must become separate source blocks.

    Return False for a single exercise, definition, theorem, proposition,
    proof, worked example, ordinary narrative, header, caption, shared
    instruction, or prescribed procedure with numbered steps. Numbered
    procedure steps are one procedure, not multiple exercises. When uncertain,
    return False.

    Return only a boolean. Answer only True or False.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Neighboring source blocks before the target, for context only.'
    )
    target_block: SourceBlockContext = dspy.InputField(
        description='The one source block being classified.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Neighboring source blocks after the target, for context only.'
    )
    contains_multiple_exercises: bool = dspy.OutputField(
        description=(
            'Return True only when the target visibly contains two or more '
            'independent numbered exercises or problems. Return False for a '
            'single exercise, prose, headers, captions, definitions, worked '
            'examples, or one multi-step procedure. Neighboring blocks are '
            'context only.'
        )
    )


class ExerciseStripRouterModule(dspy.Module):
    """Route only source blocks that visibly contain multiple exercises."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

        _set_demos(
            predictor,
            [
                dspy.Example(
                    context_before=[],
                    target_block=_context(
                        '1. first exercise 2. second exercise'
                    ),
                    context_after=[],
                    contains_multiple_exercises=True,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_context('1. Solve the system.'),
                    context_after=[],
                    contains_multiple_exercises=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_context('1. Form the matrix. 2. Reduce it.'),
                    context_after=[],
                    contains_multiple_exercises=False,
                ).with_inputs(
                    'context_before', 'target_block', 'context_after'
                ),
                dspy.Example(
                    context_before=[],
                    target_block=_context(
                        'For the following exercises, simplify each expression.'
                    ),
                    context_after=[],
                    contains_multiple_exercises=False,
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
        """Classify one target block synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.contains_multiple_exercises

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        target_block: SourceBlockContext,
        context_after: list[SourceBlockContext],
    ) -> bool:
        """Classify one target block asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            target_block=target_block,
            context_after=context_after,
        )
        return prediction.contains_multiple_exercises


__all__ = ['ExerciseStripRouterSignature', 'ExerciseStripRouterModule']
