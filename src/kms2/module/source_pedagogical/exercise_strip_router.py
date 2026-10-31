"""DSPy module for routing packed source exercises."""

import dspy

from kms2.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms2.core.model.source_pedagogical.exercise_splitter import (
    SplitRouteDecision,
)


class ExerciseStripRouterSignature(dspy.Signature):
    r"""
    Classify only the supplied `targets`; before and after lists are reference
    context. Return one indexed decision for every target, in its supplied
    position, and never classify a context block.

    Mark a target True only when it visibly packs TWO OR MORE distinct
    numbered exercises or problems that must become separate source blocks.
    Mark False for a single exercise, definition, theorem, proposition, proof,
    worked example, ordinary narrative, header, caption, shared instruction,
    or prescribed procedure with numbered steps. A shared procedure lead-in
    inside the target identifies numbered steps as one procedure, not multiple
    exercises. When uncertain, mark False.

    Return only structured decisions.
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
