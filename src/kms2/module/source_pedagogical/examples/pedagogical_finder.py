"""Curated runtime demonstrations for pedagogical finder."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.pedagogical import (
    PedagogicalBoundaryDecision,
    PedagogicalClassificationInput,
    PedagogicalStartDecision,
)


def pedagogical_start_examples() -> list[dspy.Example]:
    """Return fresh pedagogical start demonstrations."""
    return [
        dspy.Example(
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        content='# 4.6 \\textbullet Directional Derivatives and the Gradient',
                    ),
                    context_after=[],
                )
            ],
            decisions=[
                PedagogicalStartDecision(position=0, is_pedagogical_start=False)
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                PedagogicalClassificationInput(
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
                )
            ],
            decisions=[
                PedagogicalStartDecision(position=0, is_pedagogical_start=False)
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                PedagogicalClassificationInput(
                    position=0,
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
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            content='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$',
                        )
                    ],
                )
            ],
            decisions=[
                PedagogicalStartDecision(position=0, is_pedagogical_start=True)
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='280. Find the gradient of',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        content='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ ',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Then, find the gradient at point $P(1, 2).',
                        )
                    ],
                )
            ],
            decisions=[
                PedagogicalStartDecision(position=0, is_pedagogical_start=False)
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Definition 3.1. A function is continuous if ...',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Theorem 3.2. Every convergent sequence is bounded.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Procedure 1. To solve the system, perform these steps.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=3,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Worked Example 3.3. Evaluate the integral.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=4,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find each derivative.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=5,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='The function is continuous at every interior point.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=6,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Procedure 1. To solve the system, perform these steps.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='1. Substitute the value into the equation.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                PedagogicalStartDecision(position=0, is_pedagogical_start=True),
                PedagogicalStartDecision(position=1, is_pedagogical_start=True),
                PedagogicalStartDecision(position=2, is_pedagogical_start=True),
                PedagogicalStartDecision(position=3, is_pedagogical_start=True),
                PedagogicalStartDecision(
                    position=4, is_pedagogical_start=False
                ),
                PedagogicalStartDecision(
                    position=5, is_pedagogical_start=False
                ),
                PedagogicalStartDecision(
                    position=6, is_pedagogical_start=False
                ),
            ],
        ).with_inputs('items'),
    ]


def pedagogical_boundary_examples() -> list[dspy.Example]:
    """Return fresh pedagogical boundary demonstrations."""
    return [
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='280. Find the gradient of',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='For the following exercises, find the gradient.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        content='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ ',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Then, find the gradient at point $P(1, 2).',
                        )
                    ],
                )
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=False
                )
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='280. Find the gradient of',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Then, find the gradient at point $P(1, 2).',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='281. Find the gradient of',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Find the gradient of $f(x, y, z) = xy + yz + xz at point $P(1, 2, 3)$.',
                        )
                    ],
                )
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=True
                )
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='283. $f(x, y, z) = 4x^5 y^2 z^3, P(2, -1, 1), \\mathbf{u} = \\frac{1}{3}\\mathbf{i} + \\frac{2}{3}\\mathbf{j} - \\frac{2}{3}\\mathbf{k}$',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='For the following exercises, find the directional derivative of the function at point $P$ in the direction of $Q$.',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$',
                        )
                    ],
                )
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=True
                )
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='Definition 3.1. A function is continuous.',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION, content='$$f(x) = x^2$$'
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='(a) The function is differentiable.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Theorem 3.2. Every continuous function ...',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=1, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=2, is_pedagogical_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='Definition 3.1. A function is continuous at x when ...',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='A function is continuous at every point of the interval.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER, content='48'
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='The definition also applies at the interval endpoints.',
                        )
                    ],
                ),
                PedagogicalClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.HEADER,
                            content='CHAPTER 3. CONTINUITY AND DIFFERENTIABILITY',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        content='CHAPTER 3. CONTINUITY AND DIFFERENTIABILITY',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        content='3.2 Differentiability',
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='We now define the derivative of a function.',
                        )
                    ],
                ),
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=1, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=2, is_pedagogical_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                content='Procedure 1. Solve the system by performing the following steps.',
            ),
            items=[
                PedagogicalClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='Procedure 1. Solve the system by performing the following steps.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='1. Substitute the known values.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            content='1. Substitute the known values.',
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='(a) Simplify the resulting equation.',
                    ),
                    context_after=[],
                ),
                PedagogicalClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Procedure 2. Verify the solution.',
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                PedagogicalBoundaryDecision(
                    position=0, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=1, is_pedagogical_boundary=False
                ),
                PedagogicalBoundaryDecision(
                    position=2, is_pedagogical_boundary=True
                ),
            ],
        ).with_inputs('start_block', 'items'),
    ]
