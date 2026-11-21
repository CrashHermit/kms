"""Curated runtime demonstrations for exercise finder."""

import dspy

from kms.media.models import ImagePart, Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.examples.image_assets import source_image
from kms.source.processing.exercise_finder.models import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
)


def exercise_start_examples() -> list[dspy.Example]:
    """Return fresh exercise start demonstrations."""
    return [
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'For the following exercises, find the gradient.'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '280. Find the gradient of'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'For the following exercises, find the gradient.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '280. Find the gradient of'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            material=Material.from_text(
                                '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '280. Find the gradient of'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        material=Material.from_text(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '281. Find the gradient of'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=4,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '282. Evaluate each expression.'
                            ),
                        ),
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '282a. Evaluate the expression.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=False),
                ExerciseStartDecision(position=1, is_exercise_start=True),
                ExerciseStartDecision(position=2, is_exercise_start=False),
                ExerciseStartDecision(position=3, is_exercise_start=True),
                ExerciseStartDecision(position=4, is_exercise_start=False),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=Material(
                            parts=(
                                ImagePart(
                                    asset=source_image('unlabeled_triangle.png')
                                ),
                            )
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.IMAGE,
                            material=Material(
                                parts=(
                                    ImagePart(
                                        asset=source_image(
                                            'unlabeled_triangle.png'
                                        )
                                    ),
                                )
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'Find the area of the triangle.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=False),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'For the following exercises, find the derivative at P in the direction of u.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '286. $f(x, y) = -7x + 2y, P(2, -4)$'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '287. $f(x, y) = \\ln(5x + 4y)$'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '286. $f(x, y) = -7x + 2y, P(2, -4)$'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '287. $f(x, y) = \\ln(5x + 4y)$'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text('291. Find the area.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=Material(
                            parts=(
                                ImagePart(
                                    asset=source_image('unlabeled_triangle.png')
                                ),
                            )
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text('291. Find the area.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.TABLE,
                        material=Material.from_text('x | y\\n1 | 2'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text('291. Find the area.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '(a) Find the value for the second case.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=False),
                ExerciseStartDecision(position=1, is_exercise_start=False),
                ExerciseStartDecision(position=2, is_exercise_start=False),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'Activity 3.1.2. A tank drains at the rate shown in the graph. Answer the following questions about the volume of water in this tank.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'A particle moves along the given position curve. Answer parts (a) through (c) about its velocity.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '(b) Estimate the velocity at the marked time.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=Material.from_text('47'),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True),
                ExerciseStartDecision(position=1, is_exercise_start=True),
                ExerciseStartDecision(position=2, is_exercise_start=False),
                ExerciseStartDecision(position=3, is_exercise_start=False),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '3. Write the contrapositive of: If n is divisible by 6, then n is even.'
                        ),
                    ),
                    context_after=[],
                )
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True)
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '1. An integer is even exactly when it is divisible by two.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '2. If a square has equal sides then it is a rhombus, and conversely.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('1. Solve x + 1 = 3.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text('2. Solve x + 2 = 5.'),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('3. Solve x + 3 = 7.'),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('2. Solve x + 2 = 5.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text('3. Solve x + 3 = 7.'),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('4. Solve x + 4 = 9.'),
                        )
                    ],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'The truth table shows that the two statements are equivalent.'
                        ),
                    ),
                    context_after=[],
                )
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=False)
            ],
        ).with_inputs('items'),
    ]


def exercise_boundary_examples() -> list[dspy.Example]:
    """Return fresh exercise boundary demonstrations."""
    return [
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text('280. Find the gradient of'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '280. Find the gradient of'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        material=Material.from_text(
                            '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            material=Material.from_text(
                                '$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'Then, find the gradient at point P(1, 2).'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '281. Find the gradient of'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '281. Find the gradient of'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Find the gradient of f(x, y, z) = xy + yz + xz.'
                            ),
                        )
                    ],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(
                    position=0, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=1, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(position=2, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text(
                    '283. Find the directional derivative.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '283. Find the directional derivative.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'Use technology to sketch the level curve.'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '284. $f(x, y) = x^2 + 3y^2$'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Use technology to sketch the level curve.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '284. $f(x, y) = x^2 + 3y^2$'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(position=0, is_exercise_boundary=True),
                ExerciseBoundaryDecision(position=1, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text('291. Find the area.'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text('291. Find the area.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=Material(
                            parts=(
                                ImagePart(
                                    asset=source_image('unlabeled_triangle.png')
                                ),
                            )
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.IMAGE,
                            material=Material(
                                parts=(
                                    ImagePart(
                                        asset=source_image(
                                            'unlabeled_triangle.png'
                                        )
                                    ),
                                )
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.TABLE,
                        material=Material.from_text('x | y\\n1 | 2'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.TABLE,
                            material=Material.from_text('x | y\\n1 | 2'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '(a) Find the value for the second case.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                '(a) Find the value for the second case.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'The chapter continues with unrelated prose.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(
                    position=0, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=1, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=2, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(position=3, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text(
                    '12. Consider the function $h(x) = \\frac{x^2-9}{x-3}$.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=Material.from_text('47'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=Material.from_text(
                            'CHAPTER 3. LIMITS AND CONTINUITY'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=Material.from_text('3.1. THE NOTION OF LIMIT'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '(e) Sketch $y = h(x)$ and explain the behavior near $x = 3$.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=4,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            '13. Find the limit of $\\sin(x)/x$ as $x$ approaches zero.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(
                    position=0, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=1, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=2, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(
                    position=3, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(position=4, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text('14. Differentiate $p(x) = x^3$.'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=Material.from_text('3.2 The Product Rule'),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'We now develop a rule for differentiating a product of two functions.'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'Activity 3.2.1. A tank drains at the rate shown in the graph. Answer the questions about its volume.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=Material.from_text(
                            'For the following exercises, apply the product rule.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(position=0, is_exercise_boundary=True),
                ExerciseBoundaryDecision(position=1, is_exercise_boundary=True),
                ExerciseBoundaryDecision(position=2, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text(
                    '1. Construct a truth table for P if and only if Q.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            'a. List all possible truth-value pairs for P and Q.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '2. Determine when P implies Q is false.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(
                    position=0, is_exercise_boundary=False
                ),
                ExerciseBoundaryDecision(position=1, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.LIST,
                material=Material.from_text(
                    '1. An integer is even exactly when it is divisible by two.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Rewrite each of the following statements as an if and only if statement.'
                            ),
                        ),
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text(
                                '1. An integer is even exactly when it is divisible by two.'
                            ),
                        ),
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '2. If a square has equal sides then it is a rhombus, and conversely.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=Material.from_text(
                                'Rewrite each of the following statements as an if and only if statement.'
                            ),
                        ),
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text(
                                '2. If a square has equal sides then it is a rhombus, and conversely.'
                            ),
                        ),
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text(
                            '3. An integer is odd if and only if its successor is even.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(position=0, is_exercise_boundary=True),
                ExerciseBoundaryDecision(position=1, is_exercise_boundary=True),
            ],
        ).with_inputs('start_block', 'items'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.LIST,
                material=Material.from_text('1. Solve x + 1 = 3.'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('1. Solve x + 1 = 3.'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.LIST,
                        material=Material.from_text('2. Solve x + 2 = 5.'),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.LIST,
                            material=Material.from_text('3. Solve x + 3 = 7.'),
                        )
                    ],
                )
            ],
            decisions=[
                ExerciseBoundaryDecision(position=0, is_exercise_boundary=True)
            ],
        ).with_inputs('start_block', 'items'),
    ]
