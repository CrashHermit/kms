"""Few-shot examples for splitting packed exercises."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
)
from kms.module.source_pedagogical.examples.image_assets import source_image


def _context(content: str) -> SourceBlockContext:
    """Build one text-only splitter demonstration block."""
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=MultimodalContent(text=content)
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
            content_images=[],
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
            'context_before',
            'candidates',
            'reference_blocks',
            'context_after',
            'content_images',
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
            content_images=[],
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
            'context_before',
            'candidates',
            'reference_blocks',
            'context_after',
            'content_images',
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
            content_images=[],
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
            'context_before',
            'candidates',
            'reference_blocks',
            'context_after',
            'content_images',
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
                        material=MultimodalContent(
                            text='Triangle diagram for item 1.',
                            image_positions=[0],
                        ),
                    ),
                )
            ],
            context_after=[],
            content_images=[source_image('unlabeled_triangle.png')],
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
            'context_before',
            'candidates',
            'reference_blocks',
            'context_after',
            'content_images',
        ),
    ]
