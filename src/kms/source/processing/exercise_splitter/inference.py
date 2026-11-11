"""DSPy module for splitting packed source exercises."""

import dspy

from kms.source.context import (
    PositionedSourceBlock,
    SourceBlockContext,
)
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitDecision,
    SplitRouteDecision,
)


class ExerciseSplitterSignature(dspy.Signature):
    r"""
    Use ordered native image parts in the supplied blocks; never borrow images
    from another batch item.
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
    order. When whitespace separates pieces, assign that boundary whitespace
    to the preceding piece. A candidate holding only one exercise is not a
    split. Worked examples, definitions, theorems, prose, and headers are
    never split.
    Candidates and reference_blocks are a disjoint cover of the target window.
    Their original positions recover target order; reference_blocks are
    read-only and must never receive replacements.

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
    reference_blocks: list[PositionedSourceBlock] = dspy.InputField(
        description='Other target blocks with positions, for reference only.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced by candidate and reference blocks.'
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

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        reference_blocks: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
        """Split routed candidates synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=context_after,
        )
        return prediction.splits

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        reference_blocks: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
        """Split routed candidates asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=context_after,
        )
        return prediction.splits


class ExerciseStripRouterSignature(dspy.Signature):
    r"""
    Use ordered native image parts in each source block. Use only images
    referenced by these target/context blocks; never borrow from another item.
    Classify only the supplied `targets`; before and after lists are reference
    context. Return one indexed decision for every target, in its supplied
    position, and never classify a context block.

    Mark a target True only when it visibly packs TWO OR MORE distinct
    numbered exercises or problems that must become separate source blocks.
    Use ordered native image parts in supplied source blocks; never borrow
    images from other batch items. Classify only the supplied `targets`;
    before and after lists are reference context. Return one indexed decision
    for every target, in its supplied position, and never classify a context
    block.
    """

    context_before: list[SourceBlockContext] = dspy.InputField(
        description=(
            'Source blocks immediately before the targets, reference context only.'
        )
    )
    targets: list[PositionedSourceBlock] = dspy.InputField(
        description='Indexed target blocks; classify every listed block only.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description=(
            'Source blocks immediately after the targets, reference context only.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Request-local native images referenced by target and context parts.'
    )
    decisions: list[SplitRouteDecision] = dspy.OutputField(
        description=(
            'One decision for every supplied target, preserving each target '
            'position. Context blocks are reference-only and must not be returned.'
        )
    )


class ExerciseStripRouterModule(dspy.Module):
    """Route only source blocks that visibly contain multiple exercises."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitRouteDecision]:
        """Classify indexed target blocks synchronously."""
        prediction = self.predictor(
            context_before=context_before,
            targets=targets,
            context_after=context_after,
        )
        return prediction.decisions

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitRouteDecision]:
        """Classify indexed target blocks asynchronously."""
        prediction = await self.predictor.acall(
            context_before=context_before,
            targets=targets,
            context_after=context_after,
        )
        return prediction.decisions
