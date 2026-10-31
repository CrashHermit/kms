"""Curated runtime demonstrations for instruction finder."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.instruction import (
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)


def instruction_start_examples() -> list[dspy.Example]:
    """Return fresh instruction start demonstrations."""
    return [
        dspy.Example(
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find the gradient.',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='280. Find the gradient of',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='For the following exercises, find the gradient.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='280. Find the gradient of',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionStartDecision(position=0, is_instruction_start=True),
                InstructionStartDecision(
                    position=1, is_instruction_start=False
                ),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='283. $f(x, y, z) = 4x^5 y^2 z^3',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find the directional derivative.',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='284. $f(x, y) = x^2 + 3y^2',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='a. the tangent plane and b. the normal line to the surface',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='386. Find the tangent plane.',
                        )
                    ],
                ),
            ],
            decisions=[
                InstructionStartDecision(position=0, is_instruction_start=True),
                InstructionStartDecision(
                    position=1, is_instruction_start=False
                ),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find equations of',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='a. the tangent plane and b. the normal line',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='The next tasks have no exercise numbers.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Find the derivative for each unlabeled task.',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Evaluate the derivative at the given point.',
                        )
                    ],
                ),
            ],
            decisions=[
                InstructionStartDecision(position=0, is_instruction_start=True),
                InstructionStartDecision(position=1, is_instruction_start=True),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Activity 3.1.2. A tank drains at the rate shown in the graph. Answer the following questions about the volume of water in this tank.',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, estimate the rate of change.',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='A particle moves along the given position curve. Answer parts (a) through (c) about its velocity.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionStartDecision(
                    position=0, is_instruction_start=False
                ),
                InstructionStartDecision(position=1, is_instruction_start=True),
                InstructionStartDecision(
                    position=2, is_instruction_start=False
                ),
            ],
        ).with_inputs('items'),
    ]


def instruction_boundary_examples() -> list[dspy.Example]:
    """Return fresh instruction boundary demonstrations."""
    return [
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='For the following exercises, find the gradient.',
            ),
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='280. Find the gradient of',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            content='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='281. Find the gradient of',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find the gradient.',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find equations of the tangent plane.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=True
                ),
                InstructionBoundaryDecision(
                    position=1, is_instruction_boundary=True
                ),
                InstructionBoundaryDecision(
                    position=2, is_instruction_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='For the following exercises, find equations of',
            ),
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='a. the tangent plane and b. the normal line to the',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='b. the normal line to the given surface',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='a. the tangent plane and b. the normal line',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='b. the normal line to the given surface at the given point.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=False
                ),
                InstructionBoundaryDecision(
                    position=1, is_instruction_boundary=False
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='For the following exercises, find the derivative.',
            ),
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='294. Find the derivative at point (-5, -4).',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Find the derivative for the next unlabeled task.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=True
                ),
                InstructionBoundaryDecision(
                    position=1, is_instruction_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='For the following exercises, estimate the rate of change.',
            ),
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Activity 3.1.2. A tank drains at the rate shown in the graph. Answer the following questions about the volume of water in this tank.',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='A particle moves along the given position curve. Answer parts (a) through (c) about its velocity.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=True
                ),
                InstructionBoundaryDecision(
                    position=1, is_instruction_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='For the following exercises, apply the chain rule.',
            ),
            items=[
                InstructionClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.HEADER,
                            content='CHAPTER 4. DIFFERENTIATION',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        content='CHAPTER 4. DIFFERENTIATION',
                    ),
                    context_after=[],
                ),
                InstructionClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='For the following exercises, apply the chain',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER, content='48'
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='rule to each function.',
                        )
                    ],
                ),
                InstructionClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        content='4.2 The Chain Rule',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='This section introduces differentiation of composite functions.',
                        )
                    ],
                ),
            ],
            decisions=[
                InstructionBoundaryDecision(
                    position=0, is_instruction_boundary=False
                ),
                InstructionBoundaryDecision(
                    position=1, is_instruction_boundary=False
                ),
                InstructionBoundaryDecision(
                    position=2, is_instruction_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
    ]
