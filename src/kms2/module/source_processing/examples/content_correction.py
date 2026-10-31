"""Curated crop-to-transcription demonstrations for visual OCR correction."""

from pathlib import Path

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.source_processing.content_correction import (
    ContentCorrectionInput,
    ContentCorrectionOutput,
)


def content_correction_examples() -> list[dspy.Example]:
    """Return fresh indexed correction examples with packaged native crops."""
    assets = Path(__file__).with_name('assets')
    return [
        dspy.Example(
            blocks=[
                ContentCorrectionInput(
                    position=0,
                    block_type=BlockType.EQUATION,
                    content=r'$$-\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$',
                ),
                ContentCorrectionInput(
                    position=1,
                    block_type=BlockType.EQUATION,
                    content=r'$$h(x) = x^3 + 3x$$',
                ),
                ContentCorrectionInput(
                    position=2,
                    block_type=BlockType.IMAGE,
                    content='![figure](figure.png)',
                ),
            ],
            block_crops=[
                dspy.Image.from_path(str(assets / 'answer_blanks.png')),
                dspy.Image.from_path(str(assets / 'polynomial.png')),
                dspy.Image.from_path(str(assets / 'unlabeled_triangle.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    corrected_content=r'$$\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$',
                ),
                ContentCorrectionOutput(
                    position=1,
                    corrected_content=r'$$h(x) = x^2 + 3x$$',
                ),
                ContentCorrectionOutput(position=2, corrected_content=None),
            ],
        ).with_inputs('blocks', 'block_crops'),
        dspy.Example(
            blocks=[
                ContentCorrectionInput(
                    position=0,
                    block_type=BlockType.EQUATION,
                    content=r'$$q(s) = (s - 5)^9(s^2 + 4)^2$$',
                ),
            ],
            block_crops=[
                dspy.Image.from_path(str(assets / 'compact_product.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    corrected_content=r'$$q(s) = (s - 5)^2(s^2 + 4)^2$$',
                ),
            ],
        ).with_inputs('blocks', 'block_crops'),
    ]
