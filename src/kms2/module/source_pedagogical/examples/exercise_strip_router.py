"""Few-shot examples for exercise strip routing."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms2.core.model.source_pedagogical.exercise_splitter import (
    SplitRouteDecision,
)


def _context(content: str) -> SourceBlockContext:
    """Build one text-only router demonstration block."""
    return SourceBlockContext(block_type=BlockType.PARAGRAPH, content=content)


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
                    position=0,
                    source_block=_context('1. Solve the system.'),
                )
            ],
            context_after=[],
            decisions=[
                SplitRouteDecision(
                    position=0, contains_multiple_exercises=False
                )
            ],
        ).with_inputs('context_before', 'targets', 'context_after'),
    ]
