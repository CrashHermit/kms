"""Few-shot examples for exercise strip routing."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms.core.model.source_pedagogical.exercise_splitter import (
    SplitRouteDecision,
)
from kms.module.source_pedagogical.examples.image_assets import source_image


def _context(content: str) -> SourceBlockContext:
    """Build one text-only router demonstration block."""
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=MultimodalContent(text=content)
    )


def exercise_strip_router_examples() -> list[dspy.Example]:
    """Return fresh demonstrations for exercise strip routing."""
    return [
        dspy.Example(
            context_before=[_context('Earlier source material.')],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=_context(
                        'For the following exercises, simplify each expression.'
                    ),
                ),
                PositionedSourceBlock(
                    position=1,
                    source_block=_context(
                        '1. first exercise 2. second exercise'
                    ),
                ),
                PositionedSourceBlock(
                    position=2,
                    source_block=_context(
                        'To find the inverse, perform these steps: '
                        '1. Form the matrix. 2. Reduce it.'
                    ),
                ),
                PositionedSourceBlock(
                    position=3,
                    source_block=_context(
                        '8. Find the sum. 9. Find the product.'
                    ),
                ),
            ],
            context_after=[_context('Later source material.')],
            content_images=[],
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                ),
                SplitRouteDecision(
                    position=1, contains_multiple_exercises=True
                ),
                SplitRouteDecision(
                    position=2, contains_multiple_exercises=False
                ),
                SplitRouteDecision(
                    position=3, contains_multiple_exercises=True
                ),
            ],
        ).with_inputs(
            'context_before', 'targets', 'context_after', 'content_images'
        ),
        dspy.Example(
            context_before=[],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=_context('1. Solve the system.'),
                )
            ],
            context_after=[],
            content_images=[],
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                )
            ],
        ).with_inputs(
            'context_before', 'targets', 'context_after', 'content_images'
        ),
        dspy.Example(
            context_before=[
                SourceBlockContext(
                    block_type=BlockType.IMAGE,
                    material=MultimodalContent(
                        text='A diagram adjacent to the target material.',
                        image_positions=[1],
                    ),
                )
            ],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=MultimodalContent(
                            text='The target diagram.', image_positions=[0]
                        ),
                    ),
                ),
                PositionedSourceBlock(
                    position=1,
                    source_block=_context(
                        '1. Find the area. 2. Find the perimeter.'
                    ),
                ),
            ],
            context_after=[],
            content_images=[
                source_image('unlabeled_triangle.png'),
                source_image('marked_points_graph.png'),
            ],
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                ),
                SplitRouteDecision(
                    position=1, contains_multiple_exercises=True
                ),
            ],
        ).with_inputs(
            'context_before', 'targets', 'context_after', 'content_images'
        ),
    ]
