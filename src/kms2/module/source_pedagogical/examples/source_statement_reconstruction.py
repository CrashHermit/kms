"""Explicit demonstrations for statement reconstruction."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionInput,
    SourceStatementReconstructionItem,
    SourceStatementReconstructionOutput,
)


def _block(
    content: str, block_type: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    return SourceBlockContext(block_type=block_type, content=content)


def source_statement_reconstruction_examples() -> list[dspy.Example]:
    """Return demonstrations for shared directives, fragmented tasks, and figures."""
    return [
        dspy.Example(
            items=[
                SourceStatementReconstructionItem(
                    position=0,
                    request=SourceStatementReconstructionInput(
                        target_blocks=[
                            _block(
                                'Exercise 4. Find the domain of f(x)=1/(x-2).'
                            ),
                            _block('Answer: __________________'),
                        ],
                        instruction_blocks=[
                            _block(
                                'For each function, state its domain and explain excluded values.',
                                BlockType.INSTRUCTION,
                            ),
                        ],
                    ),
                ),
                SourceStatementReconstructionItem(
                    position=1,
                    request=SourceStatementReconstructionInput(
                        target_blocks=[
                            _block(
                                'Exercise 5. Find the domain of g(x)=sqrt(x+3).'
                            ),
                            _block('Answer: __________________'),
                        ],
                        instruction_blocks=[
                            _block(
                                'For each function, state its domain and explain excluded values.',
                                BlockType.INSTRUCTION,
                            ),
                        ],
                    ),
                ),
                SourceStatementReconstructionItem(
                    position=2,
                    request=SourceStatementReconstructionInput(
                        target_blocks=[
                            _block('7. (a) Solve |x-2|=5.'),
                            _block(
                                'Use the hint: |u|=v iff u=v or u=-v, for v≥0.',
                                BlockType.NOTE,
                            ),
                            _block(
                                '(b) Determine whether |x-2|=-1 has a solution; justify your answer.'
                            ),
                            _block('Name / Date — 17'),
                        ],
                    ),
                ),
                SourceStatementReconstructionItem(
                    position=3,
                    request=SourceStatementReconstructionInput(
                        target_blocks=[
                            _block(
                                '8. Refer to Figure 2. The curve rises to a peak and then falls; mark the intervals where it is increasing.'
                            ),
                            _block(
                                'Image description: A smooth curve has a local maximum near x=1 and is increasing to the left of that point.',
                                BlockType.IMAGE,
                            ),
                        ],
                    ),
                ),
            ],
            reconstructions=[
                SourceStatementReconstructionOutput(
                    position=0,
                    content='Exercise 4. Find the domain of $f(x)=\\frac{1}{x-2}$. State the domain and explain excluded values.',
                ),
                SourceStatementReconstructionOutput(
                    position=1,
                    content='Exercise 5. Find the domain of $g(x)=\\sqrt{x+3}$. State the domain and explain excluded values.',
                ),
                SourceStatementReconstructionOutput(
                    position=2,
                    content='7. (a) Solve $|x-2|=5$. Use the hint: $|u|=v$ iff $u=v$ or $u=-v$, for $v\\ge 0$. (b) Determine whether $|x-2|=-1$ has a solution; justify your answer.',
                ),
                SourceStatementReconstructionOutput(
                    position=3,
                    content='8. Refer to Figure 2. The curve rises to a peak near $x=1$ and then falls; mark the intervals where it is increasing.',
                ),
            ],
        ).with_inputs('items')
    ]
