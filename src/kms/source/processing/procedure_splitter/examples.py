"""Runtime demonstrations for procedure boundary splitting."""

import dspy

from kms.media.models import Material
from kms.source.context import PositionedSourceBlock, SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.procedure_splitter.models import (
    ProcedureSplitCandidate,
    ProcedureSplitDecision,
    ProcedureSplitPiece,
    ProcedureSplitRouteDecision,
)


def _context(content: str) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=Material.from_text(content)
    )


def procedure_splitter_examples() -> list[dspy.Example]:
    candidate = ProcedureSplitCandidate(
        position=0,
        source_block=_context(
            'Solution: 2x = 6, so x = 3. Then the chapter discusses history.'
        ),
    )
    return [
        dspy.Example(
            context_before=[],
            candidates=[candidate],
            reference_blocks=[],
            context_after=[],
            splits=[
                ProcedureSplitDecision(
                    position=0,
                    pieces=[
                        ProcedureSplitPiece(
                            content='Solution: 2x = 6, so x = 3. '
                        ),
                        ProcedureSplitPiece(
                            content='Then the chapter discusses history.'
                        ),
                    ],
                )
            ],
        ).with_inputs(
            'context_before', 'candidates', 'reference_blocks', 'context_after'
        )
    ]


def procedure_split_router_examples() -> list[dspy.Example]:
    return [
        dspy.Example(
            context_before=[],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=_context('Proof: let x be positive.'),
                )
            ],
            context_after=[],
            decisions=[
                ProcedureSplitRouteDecision(position=0, requires_split=False)
            ],
        ).with_inputs('context_before', 'targets', 'context_after'),
        dspy.Example(
            context_before=[],
            targets=[
                PositionedSourceBlock(
                    position=0,
                    source_block=_context(
                        'Solution: x = 3. The next section discusses history.'
                    ),
                )
            ],
            context_after=[],
            decisions=[
                ProcedureSplitRouteDecision(position=0, requires_split=True)
            ],
        ).with_inputs('context_before', 'targets', 'context_after'),
    ]
