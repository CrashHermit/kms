"""Shared registry helpers for runtime source workspaces."""

from kms.core.model.block import SourceBlock
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import SourceSnapshot
from kms.core.model.source_workspace import (
    ResearchWorkspace,
    SourceWorkspace,
    SourceWorkspaceBlockEntry,
    SourceWorkspaceImageEntry,
    SourceWorkspaceSourceEntry,
)
from kms.core.model.visual_asset import VisualAsset


def append_unique[T](values: list[T], value: T) -> None:
    if value not in values:
        values.append(value)


def register_block(
    workspace: SourceWorkspace,
    source: Source,
    block: SourceBlock,
    page_indexes: list[int],
) -> int:
    """Register a block once per source revision and merge its page locations."""
    key = (source.uuid, source.revision, block.uuid)
    for position, entry in enumerate(workspace.block_registry):
        if (entry.source.uuid, entry.source.revision, entry.block.uuid) == key:
            entry.source = source
            entry.block = block
            entry.page_indexes = sorted(set(entry.page_indexes + page_indexes))
            return position
    workspace.block_registry.append(
        SourceWorkspaceBlockEntry(
            source=source, page_indexes=sorted(set(page_indexes)), block=block
        )
    )
    return len(workspace.block_registry) - 1


def register_image(
    workspace: SourceWorkspace,
    asset: VisualAsset,
    source: Source,
    block_position: int,
) -> int:
    """Register an image once per source revision and retain block references."""
    key = (source.uuid, source.revision, asset.uuid)
    for position, entry in enumerate(workspace.image_registry):
        if (entry.source.uuid, entry.source.revision, entry.asset.uuid) == key:
            entry.source = source
            entry.asset = asset
            if block_position not in entry.block_positions:
                entry.block_positions.append(block_position)
            return position
    workspace.image_registry.append(
        SourceWorkspaceImageEntry(
            asset=asset, source=source, block_positions=[block_position]
        )
    )
    return len(workspace.image_registry) - 1


def register_source(
    workspace: ResearchWorkspace, snapshot: SourceSnapshot
) -> int:
    """Register one acquired snapshot and its ordered blocks and image assets."""
    source_position = next(
        (
            position
            for position, entry in enumerate(workspace.acquired_sources)
            if entry.source.uuid == snapshot.source.uuid
            and entry.source.revision == snapshot.source.revision
        ),
        None,
    )
    if source_position is None:
        workspace.acquired_sources.append(
            SourceWorkspaceSourceEntry(
                source=snapshot.source,
                page_count=len(snapshot.pages),
                block_count=len(
                    {
                        block.uuid
                        for page in snapshot.pages
                        for block in page.blocks
                    }
                ),
                image_count=len(
                    {
                        asset.uuid
                        for page in snapshot.pages
                        for block in page.blocks
                        for asset in block.assets
                    }
                ),
            )
        )
        source_position = len(workspace.acquired_sources) - 1
    for page in snapshot.pages:
        for block in page.blocks:
            block_position = register_block(
                workspace, snapshot.source, block, [page.index]
            )
            for asset in block.assets:
                register_image(
                    workspace, asset, snapshot.source, block_position
                )
    return source_position
