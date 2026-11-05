"""Contextual image-filter demonstrations with packaged native images."""

from pathlib import Path

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_processing.image_filter import (
    ImageFilterDecision,
    ImageFilterInput,
)


def image_filter_examples() -> list[dspy.Example]:
    """Demonstrate substantive figures, decoration, and instructional context."""
    assets = Path(__file__).with_name('assets')

    def context(text: str) -> list[SourceBlockContext]:
        return [
            SourceBlockContext(
                block_type=BlockType.PARAGRAPH,
                material=MultimodalContent(text=text),
            )
        ]

    return [
        dspy.Example(
            items=[
                ImageFilterInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=context(
                        'Use this diagram to compare the two quantities.'
                    ),
                    context_after=[],
                )
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'unlabeled_triangle.png'))
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ImageFilterInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=[],
                    context_after=[],
                )
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'web_ad_banner.png'))
            ],
            decisions=[ImageFilterDecision(position=0, keep=False)],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ImageFilterInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=[],
                    context_after=[],
                )
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'web_controls.png'))
            ],
            decisions=[ImageFilterDecision(position=0, keep=False)],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ImageFilterInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=context(
                        'Analyze this advertisement as an example of persuasive design.'
                    ),
                    context_after=[],
                )
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'web_ad_banner.png'))
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ImageFilterInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=context(
                        'The equation is shown in the figure.'
                    ),
                    context_after=[],
                )
            ],
            content_images=[
                dspy.Image.from_path(str(assets / 'polynomial.png'))
            ],
            decisions=[ImageFilterDecision(position=0, keep=True)],
        ).with_inputs('items', 'content_images'),
    ]
