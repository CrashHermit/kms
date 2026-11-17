"""DSPy models for detecting shared instruction support and applicability."""

import dspy

from kms.source.context import SourceBlockContext
from kms.source.processing.instruction_redistribution.models import (
    InstructionApplicabilityDecision,
    InstructionApplicabilityInput,
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)


class InstructionStartRouterSignature(dspy.Signature):
    r"""
    Classify each target independently. Mark an instruction start only for a
    shared directive or its required shared visual/table support that applies
    to multiple independent exercises. A single exercise's setup and subparts
    are not shared instructions. Include diagrams, tables, captions, or other
    visual material that functions as part of the shared directive. Return one
    indexed decision for each target; use its own context and images only.
    """

    items: list[InstructionClassificationInput] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[InstructionStartDecision] = dspy.OutputField()


class InstructionStartRouterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[InstructionClassificationInput]
    ) -> list[InstructionStartDecision]:
        return self.predictor(items=items).decisions

    async def aforward(
        self, *, items: list[InstructionClassificationInput]
    ) -> list[InstructionStartDecision]:
        return (await self.predictor.acall(items=items)).decisions


class InstructionBoundaryRouterSignature(dspy.Signature):
    r"""
    Find the exclusive end of one shared directive and its shared visual/table
    support. Keep all ordered text and required diagrams, tables, captions, and
    other support within the group. Stop before unrelated content or a new
    shared directive. Do not absorb a procedure-only support block. Return an
    indexed boundary decision for each independent target.
    """

    start_block: SourceBlockContext = dspy.InputField()
    items: list[InstructionClassificationInput] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[InstructionBoundaryDecision] = dspy.OutputField()


class InstructionBoundaryRouterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
    ) -> list[InstructionBoundaryDecision]:
        return self.predictor(start_block=start_block, items=items).decisions

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
    ) -> list[InstructionBoundaryDecision]:
        return (
            await self.predictor.acall(start_block=start_block, items=items)
        ).decisions


class InstructionApplicabilitySignature(dspy.Signature):
    r"""
    Judge whether the complete shared instruction and its native visual/table
    support apply to each complete exercise. Each item has its own context;
    other batched exercises are not evidence. Set applies only when the
    exercise is governed by the directive and needs the supplied support.
    Return one indexed decision per item; do not rewrite or solve content.
    """

    instruction_blocks: list[SourceBlockContext] = dspy.InputField()
    items: list[InstructionApplicabilityInput] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[InstructionApplicabilityDecision] = dspy.OutputField()


class InstructionApplicabilityModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionApplicabilityInput],
    ) -> list[InstructionApplicabilityDecision]:
        return self.predictor(
            instruction_blocks=instruction_blocks, items=items
        ).decisions

    async def aforward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionApplicabilityInput],
    ) -> list[InstructionApplicabilityDecision]:
        return (
            await self.predictor.acall(
                instruction_blocks=instruction_blocks, items=items
            )
        ).decisions
