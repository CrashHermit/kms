"""Native-image examples contrasting continuation with separate figures."""

from pathlib import Path

import dspy

from kms2.core.model.source_processing.image_seam import (
    ImageSeamDecision,
    ImageSeamInput,
)


def image_seam_examples() -> list[dspy.Example]:
    """Return fresh indexed positive and negative pairs in one packet."""
    assets = Path(__file__).with_name('assets')
    return [
        dspy.Example(
            items=[
                ImageSeamInput(
                    position=0, top_image_position=0, bottom_image_position=1
                ),
                ImageSeamInput(
                    position=1, top_image_position=2, bottom_image_position=3
                ),
            ],
            seam_images=[
                dspy.Image.from_path(str(assets / name))
                for name in (
                    'batch_seam_upper.png',
                    'batch_seam_lower.png',
                    'unlabeled_triangle.png',
                    'batch_separate_square.png',
                )
            ],
            decisions=[
                ImageSeamDecision(position=0, is_continuation=True),
                ImageSeamDecision(position=1, is_continuation=False),
            ],
        ).with_inputs('items', 'seam_images')
    ]
