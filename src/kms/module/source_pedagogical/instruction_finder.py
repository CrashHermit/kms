"""DSPy modules for discovering shared source instructions."""

import dspy

from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.instruction import (
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)


class InstructionStartRouterSignature(dspy.Signature):
    r"""
    Use only images referenced by each item's own target and context blocks;
    never borrow native visuals from another batched item.
    Classify each item independently. Each item's context_before and
    context_after belong only to its target_block; never use another item in
    the batch as context. Return one decision for every item, preserving its
    local position.

    Return True only when the target block is an unnumbered directive
    introducing or governing multiple exercises. Return False for a numbered
    exercise, lettered fragment without its lead-in, ordinary prose, heading,
    answer, or continuation of an earlier instruction.

    Numbered exercises remain exercises even when phrased as commands or
    preceded by a shared instruction. A single task's setup is not a shared
    instruction, even when it asks several related questions or has lettered
    subparts. A labeled activity with its own object, function, graph, or
    scenario is one exercise. Return False for that activity's lead-in; its
    subquestions do not make it a directive governing multiple independent
    exercises.
    """

    items: list[InstructionClassificationInput] = dspy.InputField(
        description=(
            'Consecutive candidates. Context lists are independently selected '
            'for each target and must not be shared across candidates.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced only by each item’s target and context.'
    )
    decisions: list[InstructionStartDecision] = dspy.OutputField(
        description='One start decision per input item, with its local position.'
    )


class InstructionStartRouterModule(dspy.Module):
    """Route source blocks that begin shared instructions in batches."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[InstructionClassificationInput],
        content_images: list[dspy.Image],
    ) -> list[InstructionStartDecision]:
        """Classify a batch of independent instruction-start candidates."""
        prediction = self.predictor(items=items, content_images=content_images)
        return prediction.decisions

    async def aforward(
        self,
        *,
        items: list[InstructionClassificationInput],
        content_images: list[dspy.Image],
    ) -> list[InstructionStartDecision]:
        """Classify a batch of instruction-start candidates asynchronously."""
        prediction = await self.predictor.acall(
            items=items, content_images=content_images
        )
        return prediction.decisions


class InstructionBoundaryRouterSignature(dspy.Signature):
    r"""
    Use only images referenced by each item's target and context plus the
    shared start block; never borrow images from another item.
    Classify each item independently relative to the shared instruction that
    begins at start_block. Each item's context_before and context_after belong
    only to its target_block; never use another item in the batch as context.
    Return one decision for every item, preserving its local position.

    Return False only when the candidate is a grammatical continuation of the
    anchored shared directive. Return True when the candidate starts an
    individual exercise, a new shared instruction, or unrelated content.

    Numbered exercises remain individual exercises even when phrased as
    commands. Apply the same boundary to unlabeled tasks; numbering is not the
    sole criterion. A candidate with its own specific expression, quantity,
    point, equation, question, or requested operation is normally an exercise,
    not instruction continuation. The candidate at a boundary is not part of
    the anchored instruction.

    A labeled activity or an unlabeled task with its own scenario, graph, or
    quantities starts an exercise, even if its lead-in asks several questions.
    It is a boundary, not a continuation of the shared directive. Page numbers
    and repeated running titles are page furniture, not semantic boundaries;
    keep scanning the contiguous instruction span across them. A genuine new
    section that introduces different material is a boundary.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one shared instruction.'
    )
    items: list[InstructionClassificationInput] = dspy.InputField(
        description=(
            'Consecutive candidates. Context lists are independently selected '
            'for each target and must not be shared across candidates.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced only by this batch’s evidence.'
    )
    decisions: list[InstructionBoundaryDecision] = dspy.OutputField(
        description='One boundary decision per input item, with its local position.'
    )


class InstructionBoundaryRouterModule(dspy.Module):
    """Route the exclusive boundary of one shared instruction in batches."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
        content_images: list[dspy.Image],
    ) -> list[InstructionBoundaryDecision]:
        """Classify a batch of instruction-boundary candidates."""
        prediction = self.predictor(
            start_block=start_block,
            items=items,
            content_images=content_images,
        )
        return prediction.decisions

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
        content_images: list[dspy.Image],
    ) -> list[InstructionBoundaryDecision]:
        """Classify a batch of boundary candidates asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block,
            items=items,
            content_images=content_images,
        )
        return prediction.decisions
