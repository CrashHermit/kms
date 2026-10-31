"""Independent seam judgments and sparse positive-only rewrite examples."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_processing.text_seam import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteOutput,
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
        tail=SourceBlockContext(block_type=kind, content=tail),
        head=SourceBlockContext(block_type=kind, content=head),
        top_context=None,
        bottom_context=None,
    )


def text_seam_judge_examples() -> list[dspy.Example]:
    """Contrast interrupted blocks with already complete neighboring sentences."""
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
    """Repair seam structure without rewriting authored content or index gaps."""
    cases = [
        (0, 'espe', 'cially', 'especially'),
        (2, 'sub-', 'graph', 'subgraph'),
        (
            4,
            'every vertex of',
            '$G$ is a vertex',
            'every vertex of $G$ is a vertex',
        ),
        (5, '$$x + y', '= 4$$', '$$x + y = 4$$'),
        (7, '$$x + y$$', '$$= 4$$', '$$x + y = 4$$'),
        (9, '$x +', 'y$', '$x + y$'),
        (
            10,
            '```python\nif p:',
            '    return p\n```',
            '```python\nif p:\n    return p\n```',
        ),
        (
            12,
            r'\begin{aligned} a &= b \\',
            r'\begin{aligned} c &= d \end{aligned}',
            r'\begin{aligned} a &= b \\ c &= d \end{aligned}',
        ),
    ]
    return [
        dspy.Example(
            items=[
                _item(position, tail, head)
                for position, tail, head, _merged in cases
            ],
            rewrites=[
                TextSeamRewriteOutput(position=position, merged=merged)
                for position, _tail, _head, merged in cases
            ],
        ).with_inputs('items')
    ]
