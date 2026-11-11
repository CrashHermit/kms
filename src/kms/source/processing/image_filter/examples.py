"""Contextual image-filter demonstrations with packaged native images."""

import dspy

from kms.media.models import Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.examples.image_assets import source_image
from kms.source.processing.image_filter.models import (
    ImageFilterDecision,
    ImageFilterInput,
)


def image_filter_examples() -> list[dspy.Example]:
    """Demonstrate substantive figures, decoration, and instructional context."""

    def context(text: str) -> list[SourceBlockContext]:
        return [
            SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=Material.from_text(text),
            )
        ]

    def item(
        asset_name: str, before: list[SourceBlockContext]
    ) -> ImageFilterInput:
        return ImageFilterInput(
            position=0,
            target=Material.from_image(source_image(asset_name)),
            context_before=before,
            context_after=[],
        )

    return [
        dspy.Example(
            items=[
                item(
                    'unlabeled_triangle.png',
                    context('Use this diagram to compare the two quantities.'),
                )
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items'),
        dspy.Example(
            items=[item('web_ad_banner.png', [])],
            decisions=[ImageFilterDecision(position=0, keep=False)],
        ).with_inputs('items'),
        dspy.Example(
            items=[item('web_controls.png', [])],
            decisions=[ImageFilterDecision(position=0, keep=False)],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                item(
                    'web_ad_banner.png',
                    context(
                        'Analyze this advertisement as an example of persuasive design.'
                    ),
                )
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items'),
        dspy.Example(
            items=[
                item(
                    'polynomial.png',
                    context('The equation is shown in the figure.'),
                )
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items'),
    ]
