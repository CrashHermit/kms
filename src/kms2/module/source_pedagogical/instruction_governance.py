"""DSPy module for governing instruction-to-statement relationships."""

import dspy

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.instruction import (
    InstructionGovernanceDecision,
    InstructionGovernanceInput,
)


class InstructionGovernanceSignature(dspy.Signature):
    r"""
    TASK
    Judge every item independently against the shared instruction_blocks.
    Each item contains one complete statement and its own reference context,
    which may clarify a lead-in, qualifier, or section boundary. Other items
    are not evidence for this item.

    governs is True when the shared directive applies to that statement;
    False when the statement is independent or belongs to another section.
    Return one decision per item, preserving its zero-based position. Return
    decisions only; do not answer, rewrite, or combine statements.
    """

    instruction_blocks: list[SourceBlockContext] = dspy.InputField(
        description='Ordered complete instruction blocks. Image descriptions appear as text; no assets or bytes.'
    )
    items: list[InstructionGovernanceInput] = dspy.InputField(
        description='Independent complete statements with candidate-specific context.'
    )
    decisions: list[InstructionGovernanceDecision] = dspy.OutputField(
        description='One indexed governance judgment for each item.'
    )


class InstructionGovernanceModule(dspy.Module):
    """Judge instruction governance for complete exercise statements."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionGovernanceInput],
    ) -> list[InstructionGovernanceDecision]:
        """Judge a packet under one shared instruction synchronously."""
        return self.predictor(
            instruction_blocks=instruction_blocks, items=items
        ).decisions

    async def aforward(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionGovernanceInput],
    ) -> list[InstructionGovernanceDecision]:
        """Judge a packet under one shared instruction asynchronously."""
        prediction = await self.predictor.acall(
            instruction_blocks=instruction_blocks, items=items
        )
        return prediction.decisions
