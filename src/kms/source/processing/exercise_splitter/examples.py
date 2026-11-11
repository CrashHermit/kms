"""Few-shot examples for splitting packed exercises."""

import dspy

from kms.media.models import Material
from kms.source.context import PositionedSourceBlock, SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.examples.image_assets import source_image
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitRouteDecision,
)


def _context(content: str) -> SourceBlockContext:
    """Build one text-only splitter demonstration block."""
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=Material.from_text(content)
    )


def exercise_splitter_examples() -> list[dspy.Example]:
    """Return fresh demonstrations for splitting packed exercises."""
    return [
        dspy.Example(
            context_before=[],
            candidates=[
                SplitCandidate(
                    position=0,
                    source_block=_context(
                        '1. first exercise 2. second exercise'
                    ),
                )
            ],
            reference_blocks=[],
            context_after=[],
            splits=[
                SplitDecision(
                    position=0,
                    pieces=[
                        SplitPiece(content='1. first exercise '),
                        SplitPiece(content='2. second exercise'),
                    ],
                )
            ],
        ).with_inputs(
            'context_before', 'candidates', 'reference_blocks', 'context_after'
        ),
        dspy.Example(
            context_before=[],
            candidates=[
                SplitCandidate(
                    position=1,
                    source_block=_context(
                        '(d) continued text 3. third exercise 4. fourth exercise'
                    ),
                )
            ],
            context_after=[],
            reference_blocks=[],
            splits=[
                SplitDecision(
                    position=1,
                    pieces=[
                        SplitPiece(content='(d) continued text '),
                        SplitPiece(content='3. third exercise '),
                        SplitPiece(content='4. fourth exercise'),
                    ],
                )
            ],
        ).with_inputs(
            'context_before', 'candidates', 'reference_blocks', 'context_after'
        ),
        dspy.Example(
            context_before=[_context('Previous source block.')],
            candidates=[
                SplitCandidate(
                    position=2,
                    source_block=_context(
                        '7. Find the mean: (a) first subpart (b) second subpart 8. next exercise'
                    ),
                ),
                SplitCandidate(
                    position=7,
                    source_block=_context(
                        '9. one exercise with (a) and (b) subparts '
                        '10. another exercise'
                    ),
                ),
            ],
            context_after=[_context('Following source block.')],
            reference_blocks=[],
            splits=[
                SplitDecision(
                    position=2,
                    pieces=[
                        SplitPiece(
                            content='7. Find the mean: (a) first subpart (b) second subpart '
                        ),
                        SplitPiece(content='8. next exercise'),
                    ],
                ),
                SplitDecision(
                    position=7,
                    pieces=[
                        SplitPiece(
                            content='9. one exercise with (a) and (b) subparts '
                        ),
                        SplitPiece(content='10. another exercise'),
                    ],
                ),
            ],
        ).with_inputs(
            'context_before', 'candidates', 'reference_blocks', 'context_after'
        ),
        dspy.Example(
            context_before=[
                _context('The following figure accompanies item 1.')
            ],
            candidates=[
                SplitCandidate(
                    position=1,
                    source_block=_context(
                        '1. Find the area. 2. Find the perimeter.'
                    ),
                )
            ],
            reference_blocks=[
                PositionedSourceBlock(
                    position=0,
                    source_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=Material.from_image(
                            source_image('unlabeled_triangle.png')
                        ),
                    ),
                )
            ],
            context_after=[],
            splits=[
                SplitDecision(
                    position=1,
                    pieces=[
                        SplitPiece(content='1. Find the area. '),
                        SplitPiece(content='2. Find the perimeter.'),
                    ],
                )
            ],
        ).with_inputs(
            'context_before', 'candidates', 'reference_blocks', 'context_after'
        ),
    ]


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
        ).with_inputs('context_before', 'targets', 'context_after'),
        dspy.Example(
            context_before=[],
            targets=[
                PositionedSourceBlock(
                    position=0, source_block=_context('1. Solve the system.')
                )
            ],
            context_after=[],
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                )
            ],
        ).with_inputs('context_before', 'targets', 'context_after'),
        dspy.Example(
            context_before=[
                SourceBlockContext(
                    block_type=BlockType.IMAGE,
                    material=Material.from_image(
                        source_image('marked_points_graph.png')
                    ),
                )
            ],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=SourceBlockContext(
                        block_type=BlockType.IMAGE,
                        material=Material.from_image(
                            source_image('unlabeled_triangle.png')
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
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                ),
                SplitRouteDecision(
                    position=1, contains_multiple_exercises=True
                ),
            ],
        ).with_inputs('context_before', 'targets', 'context_after'),
    ]
