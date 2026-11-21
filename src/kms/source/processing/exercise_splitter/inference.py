"""DSPy modules for routing and splitting exercise boundaries."""

import dspy

from kms.source.context import PositionedSourceBlock, SourceBlockContext
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitDecision,
    SplitRouteDecision,
)


class ExerciseSplitterSignature(dspy.Signature):
    r"""
    Split each candidate into ordered source fragments at exercise entry and
    exit boundaries. Include learner tasks and worked-example problems, but
    keep a task's shared setup and lettered subparts together. A list block may
    contain either continuation steps for one task or multiple independently
    answerable numbered tasks; use the wording and structure to distinguish
    them. Separate independent tasks, authored solutions/proofs, new shared
    directives, or unrelated exposition. Preserve every character exactly
    once and in order; boundary whitespace belongs to the preceding piece. Do
    not rewrite, answer, paraphrase, or repeat content. Keep structural source
    evidence and use reference blocks only for context.
    """

    context_before: list[SourceBlockContext] = dspy.InputField()
    candidates: list[SplitCandidate] = dspy.InputField()
    reference_blocks: list[PositionedSourceBlock] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    context_after: list[SourceBlockContext] = dspy.InputField()
    splits: list[SplitDecision] = dspy.OutputField()


class ExerciseSplitterModule(dspy.Module):
    """Split routed exercise-bearing blocks."""

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
        return self.predictor(
            context_before=context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=context_after,
        ).splits

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        reference_blocks: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
        prediction = await self.predictor.acall(
            context_before=context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=context_after,
        )
        return prediction.splits


class ExerciseSplitRouterSignature(dspy.Signature):
    r"""
    Classify every supplied target independently using its own native text/image
    context. Set requires_split when the block contains any boundary needed to
    isolate an exercise task from shared setup, other tasks, authored solutions,
    proofs, or unrelated exposition. A task with common setup or lettered
    subparts is one exercise. A list block may be continuation steps for one
    task or several independently answerable numbered tasks; use its content
    to distinguish them, not the list format alone. Include unnumbered tasks
    and worked-example problems. Image blocks are indivisible: always return
    false for them. Return a decision for each target position.
    """

    context_before: list[SourceBlockContext] = dspy.InputField()
    targets: list[PositionedSourceBlock] = dspy.InputField()
    context_after: list[SourceBlockContext] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[SplitRouteDecision] = dspy.OutputField()


class ExerciseSplitRouterModule(dspy.Module):
    """Route blocks needing exercise boundary splitting."""

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
        return self.predictor(
            context_before=context_before,
            targets=targets,
            context_after=context_after,
        ).decisions

    async def aforward(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitRouteDecision]:
        prediction = await self.predictor.acall(
            context_before=context_before,
            targets=targets,
            context_after=context_after,
        )
        return prediction.decisions
