"""Explicit demonstrations for procedure reconstruction."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionInput,
    SourceProcedureReconstructionItem,
    SourceProcedureReconstructionOutput,
)


def _block(
    content: str, block_type: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=block_type, material=MultimodalContent(text=content)
    )


def source_procedure_reconstruction_examples() -> list[dspy.Example]:
    """Return demonstrations preserving proof steps and algorithm conditions."""
    return [
        dspy.Example(
            items=[
                SourceProcedureReconstructionItem(
                    position=0,
                    request=SourceProcedureReconstructionInput(
                        target_blocks=[
                            _block(
                                'Proof. Since n is even, write n=2k for an integer k.'
                            ),
                            _block(
                                'Then n²=4k²=2(2k²), so n² is even. Thus the square of an even integer is even.'
                            ),
                            _block(
                                'Header: Chapter 3                                      41'
                            ),
                        ]
                    ),
                ),
                SourceProcedureReconstructionItem(
                    position=1,
                    request=SourceProcedureReconstructionInput(
                        target_blocks=[
                            _block(
                                'Start with x₀ and repeat the update xₙ₊₁=(xₙ+2/xₙ)/2.'
                            ),
                            _block(
                                'Continue until |xₙ²−2|<0.001; then stop and return xₙ.'
                            ),
                            _block('Work: _________________________________'),
                        ]
                    ),
                ),
            ],
            reconstructions=[
                SourceProcedureReconstructionOutput(
                    position=0,
                    content='Proof. Since $n$ is even, write $n=2k$ for an integer $k$. Then $n^2=4k^2=2(2k^2)$, so $n^2$ is even. Thus the square of an even integer is even.',
                ),
                SourceProcedureReconstructionOutput(
                    position=1,
                    content='Start with $x_0$ and repeat the update $x_{n+1}=\\frac{x_n+2/x_n}{2}$. Continue until $|x_n^2-2|<0.001$; then stop and return $x_n$.',
                ),
            ],
            content_images=[],
        ).with_inputs('items', 'content_images')
    ]
