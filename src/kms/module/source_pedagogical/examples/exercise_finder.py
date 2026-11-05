"""Curated runtime demonstrations for exercise finder."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.pedagogical import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
)
from kms.module.source_pedagogical.examples.image_assets import source_image


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
                        material=MultimodalContent(
                            text='For the following exercises, find the gradient.'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='280. Find the gradient of'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='For the following exercises, find the gradient.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='280. Find the gradient of'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            material=MultimodalContent(
                                text='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='280. Find the gradient of'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        material=MultimodalContent(
                            text='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='281. Find the gradient of'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=4,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='282. Evaluate each expression.'
                            ),
                        ),
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='282a. Evaluate the expression.'
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
            content_images=[],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=MultimodalContent(
                            text='A diagram presented before the exercise.',
                            image_positions=[0],
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.IMAGE,
                            material=MultimodalContent(
                                text='The same diagram supplies context.',
                                image_positions=[0],
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='Find the area of the triangle.'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=False),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
            content_images=[source_image('unlabeled_triangle.png')],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='For the following exercises, find the derivative at P in the direction of u.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='286. $f(x, y) = -7x + 2y, P(2, -4)$'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='287. $f(x, y) = \\ln(5x + 4y)$'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='286. $f(x, y) = -7x + 2y, P(2, -4)$'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='287. $f(x, y) = \\ln(5x + 4y)$'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseStartDecision(position=0, is_exercise_start=True),
                ExerciseStartDecision(position=1, is_exercise_start=True),
            ],
            content_images=[],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='291. Find the area.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=MultimodalContent(
                            text='A diagram for the exercise.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='291. Find the area.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.TABLE,
                        material=MultimodalContent(text='x | y\\n1 | 2'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='291. Find the area.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='(a) Find the value for the second case.'
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
            content_images=[],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='Activity 3.1.2. A tank drains at the rate shown in the graph. Answer the following questions about the volume of water in this tank.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='A particle moves along the given position curve. Answer parts (a) through (c) about its velocity.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='(b) Estimate the velocity at the marked time.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=MultimodalContent(text='47'),
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
            content_images=[],
        ).with_inputs('items', 'content_images'),
    ]


def exercise_boundary_examples() -> list[dspy.Example]:
    """Return fresh exercise boundary demonstrations."""
    return [
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(text='280. Find the gradient of'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='280. Find the gradient of'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.EQUATION,
                        material=MultimodalContent(
                            text='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.EQUATION,
                            material=MultimodalContent(
                                text='$$f(x, y) = \\frac{14 - x^2 - y^2}{3}.$$ '
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='Then, find the gradient at point P(1, 2).'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='281. Find the gradient of'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Then, find the gradient at point P(1, 2).'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='281. Find the gradient of'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Find the gradient of f(x, y, z) = xy + yz + xz.'
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
            content_images=[],
        ).with_inputs('start_block', 'items', 'content_images'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(
                    text='283. Find the directional derivative.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='283. Find the directional derivative.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='Use technology to sketch the level curve.'
                        ),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='284. $f(x, y) = x^2 + 3y^2$'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Use technology to sketch the level curve.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='284. $f(x, y) = x^2 + 3y^2$'
                        ),
                    ),
                    context_after=[],
                ),
            ],
            decisions=[
                ExerciseBoundaryDecision(position=0, is_exercise_boundary=True),
                ExerciseBoundaryDecision(position=1, is_exercise_boundary=True),
            ],
            content_images=[],
        ).with_inputs('start_block', 'items', 'content_images'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(text='291. Find the area.'),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='291. Find the area.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=MultimodalContent(
                            text='A diagram for exercise 291.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='A diagram for exercise 291.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.TABLE,
                        material=MultimodalContent(text='x | y\\n1 | 2'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.TABLE,
                            material=MultimodalContent(text='x | y\\n1 | 2'),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='(a) Find the value for the second case.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='(a) Find the value for the second case.'
                            ),
                        )
                    ],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='The chapter continues with unrelated prose.'
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
            content_images=[],
        ).with_inputs('start_block', 'items', 'content_images'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(
                    text='12. Consider the function $h(x) = \\frac{x^2-9}{x-3}$.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=MultimodalContent(text='47'),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=MultimodalContent(
                            text='CHAPTER 3. LIMITS AND CONTINUITY'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=MultimodalContent(
                            text='3.1. THE NOTION OF LIMIT'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=3,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='(e) Sketch $y = h(x)$ and explain the behavior near $x = 3$.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=4,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='13. Find the limit of $\\sin(x)/x$ as $x$ approaches zero.'
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
            content_images=[],
        ).with_inputs('start_block', 'items', 'content_images'),
        dspy.Example(
            start_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(
                    text='14. Differentiate $p(x) = x^3$.'
                ),
            ),
            items=[
                ExerciseClassificationInput(
                    position=0,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.HEADER,
                        material=MultimodalContent(text='3.2 The Product Rule'),
                    ),
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='We now develop a rule for differentiating a product of two functions.'
                            ),
                        )
                    ],
                ),
                ExerciseClassificationInput(
                    position=1,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='Activity 3.2.1. A tank drains at the rate shown in the graph. Answer the questions about its volume.'
                        ),
                    ),
                    context_after=[],
                ),
                ExerciseClassificationInput(
                    position=2,
                    context_before=[],
                    target_block=SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        material=MultimodalContent(
                            text='For the following exercises, apply the product rule.'
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
            content_images=[],
        ).with_inputs('start_block', 'items', 'content_images'),
    ]
