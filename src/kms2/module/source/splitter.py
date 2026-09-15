"""DSPy routing and splitting modules for packed source exercises."""

import dspy

from kms2.core.model import (
    BlockType,
    SourceBlockContext,
    SplitCandidate,
    SplitDecision,
    SplitPiece,
)


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


class ExerciseSplitterSignature(dspy.Signature):
    r"""
    Normalize routed packed source blocks for the exercise layer.

    For every supplied candidate that packs TWO OR MORE numbered exercises,
    return its target-local position and its exercises in source order. Each
    piece must contain its own leading number and complete statement text,
    copied VERBATIM: preserve wording, LaTeX, mathematics, subparts, and
    incidental markers such as a leading check mark. Do not paraphrase,
    reflow, solve, or drop text.

    Preserve a leading fragment when the candidate begins with text belonging
    to a previous exercise, such as trailing subparts before the first
    numbered exercise. Keep that fragment as the first verbatim piece without
    inventing a number, so no source text is lost.

    Every character of each candidate must land in exactly one piece, in
    order. A candidate holding only one exercise is not a split. Worked
    examples, definitions, theorems, prose, and headers are never split.

    Return one decision for each listed candidate, preserving target-local
    positions. Context blocks are reference-only and must not be returned.
    Return only structured split decisions.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Outer source blocks before the target window, for context only.'
    )
    candidates: list[SplitCandidate] = dspy.InputField(
        description='Target blocks selected for splitting, with target-local positions.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Outer source blocks after the target window, for context only.'
    )
    splits: list[SplitDecision] = dspy.OutputField(
        description=(
            'Return one decision only for each listed candidate, preserving its '
            'target-local position. Pieces must contain exact source text in '
            'source order, with no paraphrase or dropped text; context blocks '
            'must not be returned.'
        )
    )


class ExerciseSplitterModule(dspy.Module):
    """Split routed packed-exercise blocks into independent pieces."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

        _set_demos(
            predictor,
            [
                dspy.Example(
                    context_before=[],
                    candidates=[
                        SplitCandidate(
                            position=0,
                            source_block=_context(
                                '1. first exercise 2. second exercise'
                            ),
                        )
                    ],
                    context_after=[],
                    splits=[
                        SplitDecision(
                            position=0,
                            pieces=[
                                SplitPiece(content='1. first exercise'),
                                SplitPiece(content='2. second exercise'),
                            ],
                        )
                    ],
                ).with_inputs('context_before', 'candidates', 'context_after'),
                dspy.Example(
                    context_before=[],
                    candidates=[
                        SplitCandidate(
                            position=1,
                            source_block=_context(
                                '(d) continued text 3. third exercise 4. fourth exercise'
                            ),
                        )
                    ],
                    context_after=[],
                    splits=[
                        SplitDecision(
                            position=1,
                            pieces=[
                                SplitPiece(content='(d) continued text'),
                                SplitPiece(content='3. third exercise'),
                                SplitPiece(content='4. fourth exercise'),
                            ],
                        )
                    ],
                ).with_inputs('context_before', 'candidates', 'context_after'),
            ],
        )

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
        """Split routed candidates synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            candidates=candidates,
            context_after=context_after,
        )
        return prediction.splits

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
        """Split routed candidates asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            candidates=candidates,
            context_after=context_after,
        )
        return prediction.splits
