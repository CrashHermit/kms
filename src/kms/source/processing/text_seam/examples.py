"""Independent seam judgments and ordered positive-chain rewrites."""

import dspy

from kms.media.models import Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.text_seam.models import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteInput,
    TextSeamRewriteOutput,
)


def _context(
    text: str, kind: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=kind, material=Material.from_text(text)
    )


def _item(
    position: int,
    tail: str,
    head: str,
    *,
    kind: BlockType = BlockType.PARAGRAPH,
) -> TextSeamInput:
    return TextSeamInput(
        position=position,
        tail=_context(tail, kind),
        head=_context(head, kind),
        top_context=None,
        bottom_context=None,
    )


def text_seam_judge_examples() -> list[dspy.Example]:
    """Contrast interrupted blocks with complete neighboring structures."""
    return [
        dspy.Example(
            items=[
                _item(0, 'Every vertex of', '$G$ is a vertex of $H$.'),
                _item(1, 'The graph is connected.', 'The matrix is symmetric.'),
                _item(2, '$$x + y$$', '$$= 4$$', kind=BlockType.EQUATION),
            ],
            decisions=[
                TextSeamDecision(position=0, is_split=True),
                TextSeamDecision(position=1, is_split=False),
                TextSeamDecision(position=2, is_split=True),
            ],
        ).with_inputs('items')
    ]


def text_seam_rewrite_examples() -> list[dspy.Example]:
    """Rejoin faithful multi-block chains without changing authored text."""
    cases = [
        (0, ['espe', 'cially'], 'especially'),
        (2, ['sub-', 'graph'], 'subgraph'),
        (
            4,
            ['every vertex of', '$G$ is a vertex'],
            'every vertex of $G$ is a vertex',
        ),
        (5, ['$$x + y', '= 4$$'], '$$x + y = 4$$'),
        (7, ['$$x + y$$', '$$= 4$$', 'z$$'], '$$x + y = 4z$$'),
    ]
    return [
        dspy.Example(
            items=[
                TextSeamRewriteInput(
                    position=position,
                    blocks=[_context(text) for text in fragments],
                    top_context=None,
                    bottom_context=None,
                )
                for position, fragments, _merged in cases
            ],
            rewrites=[
                TextSeamRewriteOutput(position=position, merged=merged)
                for position, _fragments, merged in cases
            ],
        ).with_inputs('items')
    ]
