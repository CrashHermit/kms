"""DSPy routing and splitting modules for packed source exercises."""

import dspy

from kms2.core.model import (
    SourceBlockContext,
    SplitCandidate,
    SplitDecision,
)


class ExerciseStripRouterSignature(dspy.Signature):
    """Return true only for packed numbered exercises, never ordinary prose."""

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

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ExerciseStripRouterSignature)
        self.predictor.set_lm(language_model)

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
    """Partition routed packed source blocks into exact ordered text pieces."""

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

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ExerciseSplitterSignature)
        self.predictor.set_lm(language_model)

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
