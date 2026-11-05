"""Native-image demonstrations of independent logical image groups."""

from pathlib import Path

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionOutput,
)


def image_description_examples() -> list[dspy.Example]:
    """Keep logical groups, graph details, and contextual labels independent."""
    assets = Path(__file__).with_name('assets')
    return [
        dspy.Example(
            items=[
                ImageDescriptionInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0, 1]),
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='The graph continues across the page break.'
                            ),
                        )
                    ],
                    context_after=[],
                ),
                ImageDescriptionInput(
                    position=1,
                    target=MultimodalContent(text=None, image_positions=[2]),
                    context_before=[],
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Describe the polygon without assigning missing dimensions.'
                            ),
                        )
                    ],
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / name))
                for name in (
                    'batch_seam_upper.png',
                    'batch_seam_lower.png',
                    'unlabeled_triangle.png',
                )
            ],
            descriptions=[
                ImageDescriptionOutput(
                    position=0,
                    description='A rectangular grid with one continuous diagonal black stroke running from near the upper left to near the lower right across both ordered parts.',
                ),
                ImageDescriptionOutput(
                    position=1,
                    description='An unlabeled triangular outline. No side lengths or angle measurements are shown.',
                ),
            ],
        ).with_inputs('items', 'content_images'),
        dspy.Example(
            items=[
                ImageDescriptionInput(
                    position=0,
                    target=MultimodalContent(text=None, image_positions=[0]),
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='The horizontal axis x is elapsed time in seconds; the vertical axis y is distance in metres.'
                            ),
                        )
                    ],
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.PARAGRAPH,
                            material=MultimodalContent(
                                text='Use the graph to examine the behavior near x=1.'
                            ),
                        )
                    ],
                ),
                ImageDescriptionInput(
                    position=1,
                    target=MultimodalContent(text=None, image_positions=[1]),
                    context_before=[
                        SourceBlockContext(
                            block_type=BlockType.CAPTION,
                            material=MultimodalContent(
                                text='This image is panel (b): blank axes for plotting y=k(x).'
                            ),
                        )
                    ],
                    context_after=[],
                ),
                ImageDescriptionInput(
                    position=2,
                    target=MultimodalContent(text=None, image_positions=[2]),
                    context_before=[],
                    context_after=[
                        SourceBlockContext(
                            block_type=BlockType.CAPTION,
                            material=MultimodalContent(
                                text='The figure contains axes for f in panel (a) and g in panel (b); this crop has no panel label.'
                            ),
                        )
                    ],
                ),
            ],
            content_images=[
                dspy.Image.from_path(str(assets / name))
                for name in (
                    'marked_points_graph.png',
                    'blank_plot_axes.png',
                    'blank_plot_axes.png',
                )
            ],
            descriptions=[
                ImageDescriptionOutput(
                    position=0,
                    description=(
                        'A coordinate grid with horizontal $x$ (elapsed time in seconds) '
                        'and vertical $y$ (distance in metres). Connected line '
                        'segments rise to an open point explicitly annotated '
                        '$(2,15)$, then descend. A separate filled point is '
                        'explicitly annotated $(2,5)$.'
                    ),
                ),
                ImageDescriptionOutput(
                    position=1,
                    description='Blank coordinate axes on a grid, with no plotted curve. The supplied caption identifies this crop as panel (b), for plotting $y=k(x)$.',
                ),
                ImageDescriptionOutput(
                    position=2,
                    description='Blank coordinate axes on a grid, with no plotted curve or visible panel identifier. The caption does not identify which panel this crop is.',
                ),
            ],
        ).with_inputs('items', 'content_images'),
    ]
