"""DSPy modules for discovering exercise components."""

import dspy

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.pedagogical import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
)


class ExerciseStartRouterSignature(dspy.Signature):
    r"""
    Classify every indexed target independently. Each item's before and after
    lists are its own context; never use another item in `items` as context.
    Return one decision for every input position, preserving those positions.

    Mark a target True only when it begins one distinct individual exercise.
    Adjacent exercises are separate even when they share a topic or shared
    instructions. Do not mark shared instructions, continuations, attached
    equations, images, tables, lists, explanatory blocks, or subparts as starts.

    An individual exercise may be labeled as an activity or have no number.
    Its setup can introduce a graph, function, object, or scenario and ask
    several related questions. Mark that setup True: lettered subparts under
    one setup are one exercise, not separate exercises or a shared instruction.
    A shared instruction instead governs multiple independent exercises.

    Page numbers, running headers, and section headings are not exercise starts.

    Return decisions only; do not include reasons, spans, or text.
    """

    items: list[ExerciseClassificationInput] = dspy.InputField(
        description=(
            'Indexed targets with independently selected context for each item; '
            'items are not context for one another.'
        )
    )
    decisions: list[ExerciseStartDecision] = dspy.OutputField(
        description='One indexed start decision for every input item.'
    )


class ExerciseStartRouterModule(dspy.Module):
    """Route batches of source blocks that begin exercise components."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[ExerciseClassificationInput]
    ) -> list[ExerciseStartDecision]:
        """Classify a batch of independent exercise-start candidates."""
        prediction = self.predictor(items=items)
        return prediction.decisions

    async def aforward(
        self, *, items: list[ExerciseClassificationInput]
    ) -> list[ExerciseStartDecision]:
        """Classify a batch of exercise-start candidates asynchronously."""
        prediction = await self.predictor.acall(items=items)
        return prediction.decisions


class ExerciseBoundaryRouterSignature(dspy.Signature):
    r"""
    Classify each indexed target relative to `start_block`. Each item's context
    lists are independent; do not use another item as context. Return one
    decision for every input position, preserving those positions.

    Mark False when the exercise continues through the target, including its
    equations, images, tables, lists, explanatory blocks, and subparts.

    A page number or repeated running chapter/section title is page furniture,
    not a semantic boundary. Mark False so scanning continues across it to the
    rest of the same exercise. Such layout blocks remain inside the contiguous
    span; False does not claim that they are mathematical exercise content.

    Mark True for the start of another exercise (including a labeled activity
    or an unlabeled independent task), a new shared instruction, a genuine new
    section introducing different material, or unrelated exposition. Do not
    treat every header as a new section: distinguish running titles/page numbers
    from headings that introduce new subject matter. A new numbered exercise
    is a boundary even when its topic is similar.

    The boundary is exclusive: a True target is not part of this exercise
    and must not be consumed.

    Return decisions only; do not include reasons, spans, or text.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one individual exercise.'
    )
    items: list[ExerciseClassificationInput] = dspy.InputField(
        description=(
            'Indexed boundary targets with independently selected context for '
            'each item; items are not context for one another.'
        )
    )
    decisions: list[ExerciseBoundaryDecision] = dspy.OutputField(
        description='One indexed exclusive-boundary decision for every input item.'
    )


class ExerciseBoundaryRouterModule(dspy.Module):
    """Route batches of exclusive exercise-boundary candidates."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ExerciseClassificationInput],
    ) -> list[ExerciseBoundaryDecision]:
        """Classify a batch of exercise-boundary candidates."""
        prediction = self.predictor(start_block=start_block, items=items)
        return prediction.decisions

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ExerciseClassificationInput],
    ) -> list[ExerciseBoundaryDecision]:
        """Classify a batch of boundaries asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block, items=items
        )
        return prediction.decisions
