"""Native-image examples contrasting continuation with separate figures."""

import dspy

from kms.media.models import Material
from kms.source.processing.examples.image_assets import source_image
from kms.source.processing.image_seam.models import (
    ImageSeamDecision,
    ImageSeamInput,
)


def image_seam_examples() -> list[dspy.Example]:
    """Return fresh positive and negative pairs in one packet."""
    assets = [
        source_image(name)
        for name in (
            'batch_seam_upper.png',
            'batch_seam_lower.png',
            'unlabeled_triangle.png',
            'batch_separate_square.png',
        )
    ]
    return [
        dspy.Example(
            items=[
                ImageSeamInput(
                    position=0,
                    top=Material.from_image(assets[0]),
                    bottom=Material.from_image(assets[1]),
                ),
                ImageSeamInput(
                    position=1,
                    top=Material.from_image(assets[2]),
                    bottom=Material.from_image(assets[3]),
                ),
            ],
            decisions=[
                ImageSeamDecision(position=0, is_continuation=True),
                ImageSeamDecision(position=1, is_continuation=False),
            ],
        ).with_inputs('items')
    ]
