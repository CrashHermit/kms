"""Curated crop-to-transcription demonstrations for visual OCR correction."""

from pathlib import Path

import dspy
from PIL import Image

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.source_processing.content_correction import (
    ContentCorrectionBlockContext,
    ContentCorrectionInput,
    ContentCorrectionOutput,
)


def _block_context(
    assets: Path, name: str, image_position: int
) -> ContentCorrectionBlockContext:
    with Image.open(assets / name) as image:
        width, height = image.size
    return ContentCorrectionBlockContext(
        image_position=image_position,
        target_bounds=(0, 0, width, height),
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
                    material=MultimodalContent(
                        text=r'$$-\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$',
                        image_positions=[0],
                    ),
                    context=_block_context(assets, 'answer_blanks.png', 0),
                ),
                ContentCorrectionInput(
                    position=1,
                    block_type=BlockType.EQUATION,
                    material=MultimodalContent(
                        text=r'$$h(x) = x^3 + 3x$$', image_positions=[1]
                    ),
                    context=_block_context(assets, 'polynomial.png', 1),
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'answer_blanks.png')),
                dspy.Image.from_path(str(assets / 'polynomial.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=True,
                    corrected_content=r'$$\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$',
                    cleanup_instructions=[],
                ),
                ContentCorrectionOutput(
                    position=1,
                    keep=True,
                    corrected_content=r'$$h(x) = x^2 + 3x$$',
                    cleanup_instructions=[],
                ),
            ],
        ).with_inputs('blocks', 'content_images'),
        dspy.Example(
            blocks=[
                ContentCorrectionInput(
                    position=0,
                    block_type=BlockType.EQUATION,
                    material=MultimodalContent(
                        text=r'$$q(s) = (s - 5)^9(s^2 + 4)^2$$',
                        image_positions=[0],
                    ),
                    context=_block_context(assets, 'compact_product.png', 0),
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'compact_product.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=True,
                    corrected_content=r'$$q(s) = (s - 5)^2(s^2 + 4)^2$$',
                    cleanup_instructions=[],
                ),
            ],
        ).with_inputs('blocks', 'content_images'),
        dspy.Example(
            blocks=[
                ContentCorrectionInput(
                    position=0,
                    block_type=BlockType.PARAGRAPH,
                    material=MultimodalContent(
                        text='Image  Video',
                        image_positions=[0],
                    ),
                    context=_block_context(assets, 'web_controls.png', 0),
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'web_controls.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=False,
                    corrected_content=None,
                    cleanup_instructions=[],
                ),
            ],
        ).with_inputs('blocks', 'content_images'),
        dspy.Example(
            blocks=[
                ContentCorrectionInput(
                    position=0,
                    block_type=BlockType.PARAGRAPH,
                    material=MultimodalContent(
                        text=(
                            'For f(x) = x^2, the derivative is 2x. '
                            'ADVERTISEMENT - Subscribe now: https://exam ple .com/offer'
                        ),
                        image_positions=[0],
                    ),
                    context=_block_context(assets, 'mixed_math_ad.png', 0),
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'mixed_math_ad.png')),
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=True,
                    corrected_content=(
                        'For f(x) = x^2, the derivative is 2x. '
                        'ADVERTISEMENT - Subscribe now: https://exam ple .com/offer'
                    ),
                    cleanup_instructions=[
                        "Remove the entire footer beginning 'ADVERTISEMENT - Subscribe now:'; it is a promotional artifact, including its corrupted URL fragment."
                    ],
                ),
            ],
        ).with_inputs('blocks', 'content_images'),
    ]
