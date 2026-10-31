"""DSPy module for splitting packed source exercises."""

import dspy

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitDecision,
)


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
    order. When whitespace separates pieces, assign that boundary whitespace
    to the preceding piece. A candidate holding only one exercise is not a
    split. Worked examples, definitions, theorems, prose, and headers are
    never split.

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
