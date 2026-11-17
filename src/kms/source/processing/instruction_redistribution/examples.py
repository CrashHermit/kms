"""Runtime demonstrations for instruction redistribution."""

import dspy

from kms.media.models import Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.instruction_redistribution.models import (
    InstructionApplicabilityDecision,
    InstructionApplicabilityInput,
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)


def _context(content: str) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=Material.from_text(content)
    )


def instruction_start_examples() -> list[dspy.Example]:
    item = InstructionClassificationInput(
        position=0,
        context_before=[],
        target_block=_context('For each problem, use the diagram.'),
        context_after=[],
    )
    return [
        dspy.Example(
            items=[item],
            decisions=[
                InstructionStartDecision(position=0, is_instruction_start=True)
            ],
        ).with_inputs('items')
    ]


def instruction_boundary_examples() -> list[dspy.Example]:
    start = _context('For each problem, use the diagram.')
    item = InstructionClassificationInput(
        position=0,
        context_before=[],
        target_block=_context('Find the area of the shaded region.'),
        context_after=[],
    )
    return [
        dspy.Example(
            start_block=start,
            items=[item],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=True
                )
            ],
        ).with_inputs('start_block', 'items')
    ]


def instruction_applicability_examples() -> list[dspy.Example]:
    items = [
        InstructionApplicabilityInput(
            position=0,
            context_before=[],
            exercise_blocks=[
                _context('Find the area of the shaded region in the diagram.')
            ],
            context_after=[],
        ),
        InstructionApplicabilityInput(
            position=1,
            context_before=[],
            exercise_blocks=[
                _context('Explain why every prime greater than two is odd.')
            ],
            context_after=[],
        ),
    ]
    return [
        dspy.Example(
            instruction_blocks=[
                _context('For the geometry problems, use the diagram.')
            ],
            items=items,
            decisions=[
                InstructionApplicabilityDecision(position=0, applies=True),
                InstructionApplicabilityDecision(position=1, applies=False),
            ],
        ).with_inputs('instruction_blocks', 'items')
    ]
