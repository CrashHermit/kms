"""Project canonical source windows into request-local model context."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.tokenization.text import count_text_tokens
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.source import BlockType, SourceBlock


class SourceBlockContext(BaseModel):
    """Projected source-block data for downstream model inputs."""

    block_type: BlockType
    material: Material


class PositionedSourceBlock(BaseModel):
    """One source block with a zero-based position in a request target."""

    position: int = Field(ge=0, strict=True)
    source_block: SourceBlockContext


class SourceBlockWindow(BaseModel):
    """Selected canonical source blocks retained until model projection."""

    context_before: list[SourceBlock] = Field(default_factory=list)
    target: list[SourceBlock] = Field(default_factory=list)
    context_after: list[SourceBlock] = Field(default_factory=list)


class SourceContextWindow(BaseModel):
    """Ordered model-facing source context split around a target span."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target: list[SourceBlockContext] = Field(default_factory=list)
    context_after: list[SourceBlockContext] = Field(default_factory=list)


def count_source_tokens(
    blocks: Sequence[SourceBlock],
    counters: Sequence[MaterialTokenEstimator],
) -> list[int]:
    """Count source text and projected native images for ordered blocks."""
    if not blocks:
        return []
    text_counts = count_text_tokens(
        [block.content for block in blocks], counters
    )
    image_estimator = counters[0]
    image_counts = [
        image_estimator.estimate_images(
            project_source_block(block).material.images
        )
        for block in blocks
    ]
    return [
        text_count + image_count
        for text_count, image_count in zip(
            text_counts, image_counts, strict=True
        )
    ]


def project_source_block(block: SourceBlock) -> SourceBlockContext:
    """Project source text followed by its canonical image, when present."""
    parts: list[TextPart | ImagePart] = []
    if block.content is not None:
        parts.append(TextPart(text=block.content))
    if block.image_path is not None:
        parts.append(
            ImagePart(
                asset=ImageAsset(
                    ('path', block.image_path), Path(block.image_path)
                )
            )
        )
    return SourceBlockContext(
        block_type=block.block_type,
        material=Material(parts=tuple(parts)),
    )


def project_source_material(
    content: str,
    blocks: Sequence[SourceBlock],
) -> Material:
    """Project annotation text followed by images from ordered evidence."""
    return Material(
        parts=(
            TextPart(text=content),
            *(
                ImagePart(
                    asset=ImageAsset(
                        ('path', block.image_path), Path(block.image_path)
                    )
                )
                for block in blocks
                if block.image_path is not None
            ),
        )
    )


def project_source_window(window: SourceBlockWindow) -> SourceContextWindow:
    """Project blocks in before/target/after encounter order."""
    return SourceContextWindow(
        context_before=[
            project_source_block(block) for block in window.context_before
        ],
        target=[project_source_block(block) for block in window.target],
        context_after=[
            project_source_block(block) for block in window.context_after
        ],
    )


def select_window(
    blocks: list[SourceBlock],
    target_positions: list[int],
    *,
    token_counts: Sequence[int],
    backward_budget: int | None = None,
    forward_budget: int | None = None,
    target_budget: int = 0,
    max_targets: int | None = None,
) -> SourceBlockWindow:
    """Select canonical blocks and budgeted context around target positions."""
    target = [blocks[position] for position in target_positions]
    target_start = target_positions[0]
    target_end = target_positions[-1]
    if target_budget > 0:
        target_cost = sum(
            token_counts[position] for position in target_positions
        )
        for position in range(target_end + 1, len(blocks)):
            if max_targets is not None and len(target) >= max_targets:
                break
            cost = token_counts[position]
            if target_cost + cost > target_budget:
                break
            target.append(blocks[position])
            target_cost += cost
            target_end = position

    context_before: list[SourceBlock] = []
    if backward_budget is not None and backward_budget > 0:
        before_cost = 0
        for position in range(target_start - 1, -1, -1):
            block = blocks[position]
            cost = token_counts[position]
            if before_cost + cost > backward_budget:
                break
            context_before.append(block)
            before_cost += cost
        context_before.reverse()

    context_after: list[SourceBlock] = []
    if forward_budget is not None and forward_budget > 0:
        after_cost = 0
        for position in range(target_end + 1, len(blocks)):
            block = blocks[position]
            cost = token_counts[position]
            if after_cost + cost > forward_budget:
                break
            context_after.append(block)
            after_cost += cost

    return SourceBlockWindow(
        context_before=context_before,
        target=target,
        context_after=context_after,
    )
