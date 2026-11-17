"""DSPy modules for finding complete authored procedures."""

import dspy

from kms.source.context import SourceBlockContext
from kms.source.processing.procedure_finder.models import (
    ProcedureBoundaryDecision,
    ProcedureClassificationInput,
    ProcedureStartDecision,
)


class ProcedureStartRouterSignature(dspy.Signature):
    r"""
    Classify each target with only its own context and native images. Mark a
    start for an authored complete method, worked solution, calculation,
    derivation, or proof. A learner-facing exercise that asks the learner to
    describe or find a method is an exercise, not an authored procedure. Do not
    classify standalone steps or incidental equations as starts. Return one
    position-addressed decision for every independent item.
    """

    items: list[ProcedureClassificationInput] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[ProcedureStartDecision] = dspy.OutputField()


class ProcedureStartRouterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[ProcedureClassificationInput]
    ) -> list[ProcedureStartDecision]:
        return self.predictor(items=items).decisions

    async def aforward(
        self, *, items: list[ProcedureClassificationInput]
    ) -> list[ProcedureStartDecision]:
        return (await self.predictor.acall(items=items)).decisions


class ProcedureBoundaryRouterSignature(dspy.Signature):
    r"""
    For each independently contextualized target, mark a boundary at the
    beginning of unrelated exposition, the next distinct procedure, or an
    exercise task. Keep all steps, intermediate calculations, equations,
    diagrams and explanatory text belonging to the current complete method,
    solution, derivation, or proof together. The True boundary block is
    exclusive and is not part of the current procedure. Return one decision
    for every input position.
    """

    start_block: SourceBlockContext = dspy.InputField()
    items: list[ProcedureClassificationInput] = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    decisions: list[ProcedureBoundaryDecision] = dspy.OutputField()


class ProcedureBoundaryRouterModule(dspy.Module):
    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ProcedureClassificationInput],
    ) -> list[ProcedureBoundaryDecision]:
        return self.predictor(start_block=start_block, items=items).decisions

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ProcedureClassificationInput],
    ) -> list[ProcedureBoundaryDecision]:
        return (
            await self.predictor.acall(start_block=start_block, items=items)
        ).decisions
