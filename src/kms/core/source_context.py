"""Project canonical source windows into request-local model context."""

from collections.abc import Sequence
from pathlib import Path

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.model.block import SourceBlock
from kms.core.model.context import (
    SourceBlockContext,
    SourceBlockWindow,
    SourceContextWindow,
)
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.tokenization.text import count_text_tokens


def count_source_tokens(
    blocks: Sequence[SourceBlock],
    counters: Sequence[MaterialTokenEstimator],
) -> list[int]:
    """Count source text and projected native assets for ordered blocks."""
    if not blocks:
        return []
    text_counts = count_text_tokens(
        [block.content for block in blocks], counters
    )
    image_estimator = counters[0]
    image_counts = []
    # shortcut: repeated paths count per block; deduplicate windows if they limit context.
    for block in blocks:
        builder = MultimodalContentBuilder()
        project_source_block(block, builder)
        image_counts.append(image_estimator.estimate_images(builder.images))
    return [
        text_count + image_count
        for text_count, image_count in zip(
            text_counts, image_counts, strict=True
        )
    ]


def project_source_block(
    block: SourceBlock, builder: MultimodalContentBuilder
) -> SourceBlockContext:
    """Project a source block into the prediction's shared image catalog."""
    return SourceBlockContext(
        block_type=block.block_type,
        material=builder.build(
            text=block.content,
            images=[
                ContentImage(('path', asset.path), Path(asset.path))
                for asset in block.assets
            ],
        ),
    )


def project_source_window(
    window: SourceBlockWindow, builder: MultimodalContentBuilder
) -> SourceContextWindow:
    """Project blocks in before/target/after encounter order."""
    return SourceContextWindow(
        context_before=[
            project_source_block(block, builder)
            for block in window.context_before
        ],
        target=[
            project_source_block(block, builder) for block in window.target
        ],
        context_after=[
            project_source_block(block, builder)
            for block in window.context_after
        ],
    )
