"""Curated crop-to-transcription demonstrations for visual OCR correction."""

from pathlib import Path

import dspy
from PIL import Image

from kms.media.models import ImagePart, Material, TextPart
from kms.source.models.source import BlockType
from kms.source.processing.correction.models import (
    ContentCorrectionBlockContext,
    ContentCorrectionInput,
    ContentCorrectionOutput,
)
from kms.source.processing.examples.image_assets import (
    ASSETS_DIRECTORY,
    source_image,
)


def _block_context(assets: Path, name: str) -> ContentCorrectionBlockContext:
    with Image.open(assets / name) as image:
        width, height = image.size
    return ContentCorrectionBlockContext(
        image=source_image(name), target_bounds=(0, 0, width, height)
    )


def _input(
    position: int,
    block_type: BlockType,
    text: str,
    name: str,
) -> ContentCorrectionInput:
    return ContentCorrectionInput(
        position=position,
        block_type=block_type,
        material=Material(
            parts=(TextPart(text=text), ImagePart(asset=source_image(name)))
        ),
        context=_block_context(ASSETS_DIRECTORY, name),
    )


def content_correction_examples() -> list[dspy.Example]:
    """Return fresh correction examples with packaged native crops."""
    return [
        dspy.Example(
            blocks=[
                _input(
                    0,
                    BlockType.EQUATION,
                    r'$$-\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$',
                    'answer_blanks.png',
                ),
                _input(
                    1,
                    BlockType.EQUATION,
                    r'$$h(x) = x^3 + 3x$$',
                    'polynomial.png',
                ),
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
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                _input(
                    0,
                    BlockType.EQUATION,
                    r'$$q(s) = (s - 5)^9(s^2 + 4)^2$$',
                    'compact_product.png',
                )
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=True,
                    corrected_content=r'$$q(s) = (s - 5)^2(s^2 + 4)^2$$',
                    cleanup_instructions=[],
                )
            ],
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                _input(
                    0, BlockType.PARAGRAPH, 'Image  Video', 'web_controls.png'
                )
            ],
            corrected_blocks=[
                ContentCorrectionOutput(
                    position=0,
                    keep=False,
                    corrected_content=None,
                    cleanup_instructions=[],
                )
            ],
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                _input(
                    0,
                    BlockType.PARAGRAPH,
                    'For f(x) = x^2, the derivative is 2x. '
                    'ADVERTISEMENT - Subscribe now: https://exam ple .com/offer',
                    'mixed_math_ad.png',
                )
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
                )
            ],
        ).with_inputs('blocks'),
    ]
