"""DSPy modules for routing and splitting procedure boundaries."""

import dspy

from kms.source.context import PositionedSourceBlock, SourceBlockContext
from kms.source.processing.procedure_splitter.models import (
    ProcedureSplitCandidate,
    ProcedureSplitDecision,
    ProcedureSplitRouteDecision,
)


class ProcedureSplitterSignature(dspy.Signature):
    r"""
    Split each candidate into ordered source fragments at procedure entry and
    exit boundaries. Procedures include complete methods, worked solutions,
    calculations, derivations, and proofs. Separate distinct procedures and
    retain leading/trailing unrelated source content outside them. Preserve
    every character exactly once and in order; boundary whitespace belongs to
    the preceding piece. Do not paraphrase, solve, add steps, or rewrite.
    Preserve structural context and use reference blocks only as context.
    """

    context_before: list[SourceBlockContext] = dspy.InputField()
    candidates: list[ProcedureSplitCandidate] = dspy.InputField()
    reference_blocks: list[PositionedSourceBlock] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    context_after: list[SourceBlockContext] = dspy.InputField()
    splits: list[ProcedureSplitDecision] = dspy.OutputField()


class ProcedureSplitterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[ProcedureSplitCandidate],
        reference_blocks: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[ProcedureSplitDecision]:
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
        candidates: list[ProcedureSplitCandidate],
        reference_blocks: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[ProcedureSplitDecision]:
        prediction = await self.predictor.acall(
            context_before=context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=context_after,
        )
        return prediction.splits


class ProcedureSplitRouterSignature(dspy.Signature):
    r"""
    Classify each target independently with its own native visual and textual
    context. Set requires_split when the target contains any procedure entry or
    exit needed to separate a complete method, solution, calculation,
    derivation, or proof from unrelated content or another procedure. An
    exercise prompt asking the reader to describe a method is not itself a
    procedure. Image blocks are indivisible: always return false for them.
    Return one decision for every target position.
    """

    context_before: list[SourceBlockContext] = dspy.InputField()
    targets: list[PositionedSourceBlock] = dspy.InputField()
    context_after: list[SourceBlockContext] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[ProcedureSplitRouteDecision] = dspy.OutputField()


class ProcedureSplitRouterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[ProcedureSplitRouteDecision]:
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
    ) -> list[ProcedureSplitRouteDecision]:
        prediction = await self.predictor.acall(
            context_before=context_before,
            targets=targets,
            context_after=context_after,
        )
        return prediction.decisions
