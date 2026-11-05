"""Read owned source material into runtime workspace registries."""

from collections.abc import Sequence

from kms.core.model.source_evidence import (
    SourceImage,
    SourcePassage,
    SourceReadResult,
)
from kms.core.model.source_workspace import (
    SourceWorkspace,
    SourceWorkspaceBlockEntry,
    SourceWorkspaceImageEntry,
)
from kms.core.source_workspace import register_block, register_image
from kms.service.source_evidence import SourceEvidenceService


async def read_passages(
    source_evidence: SourceEvidenceService,
    user_uuid: str,
    entries: Sequence[SourceWorkspaceBlockEntry],
    *,
    expand_memberships: bool,
    before: int,
    after: int,
) -> list[SourcePassage]:
    """Read every requested source block from its owner's graph revision."""
    grouped: dict[tuple[str, int], list[SourceWorkspaceBlockEntry]] = {}
    for entry in entries:
        grouped.setdefault(
            (entry.source.uuid, entry.source.revision), []
        ).append(entry)
    passages: list[SourcePassage] = []
    for (source_uuid, revision), group in grouped.items():
        block_uuids = list(dict.fromkeys(entry.block.uuid for entry in group))
        source_passages = await source_evidence.read_passages(
            user_uuid,
            block_uuids,
            expand_memberships=expand_memberships,
            before=before,
            after=after,
        )
        selected = {
            block_uuid
            for passage in source_passages
            for block_uuid in passage.selected_block_uuids
        }
        if any(
            passage.source.uuid != source_uuid
            or passage.source.revision != revision
            for passage in source_passages
        ) or not set(block_uuids).issubset(selected):
            raise ValueError('Requested source revision is no longer available')
        passages.extend(source_passages)
    return passages


async def read_images(
    source_evidence: SourceEvidenceService,
    user_uuid: str,
    entries: Sequence[SourceWorkspaceImageEntry],
) -> list[SourceImage]:
    """Read original images only from their owning graph revisions."""
    return await source_evidence.read_images_at_revisions(
        user_uuid,
        [
            (entry.source.uuid, entry.source.revision, entry.asset.uuid)
            for entry in entries
        ],
    )


class SourceWorkspaceService:
    """Read registered source material from its owning graph."""

    def __init__(self, source_evidence: SourceEvidenceService) -> None:
        self._source_evidence = source_evidence

    async def read_blocks(
        self,
        user_uuid: str,
        workspace: SourceWorkspace,
        block_positions: list[int],
        *,
        before: int = 0,
        after: int = 0,
        expand_memberships: bool = False,
        load_images: bool = True,
    ) -> SourceReadResult:
        if not block_positions:
            return SourceReadResult()
        entries = [
            workspace.block_registry[position] for position in block_positions
        ]
        passages = await read_passages(
            self._source_evidence,
            user_uuid,
            entries,
            expand_memberships=expand_memberships,
            before=before,
            after=after,
        )
        block_positions_read: list[int] = []
        image_positions_read: list[int] = []
        for passage in passages:
            for item in passage.blocks:
                position = register_block(
                    workspace, passage.source, item.block, item.page_indexes
                )
                if position not in block_positions_read:
                    block_positions_read.append(position)
                for asset in item.block.assets:
                    image_position = register_image(
                        workspace, asset, passage.source, position
                    )
                    if image_position not in image_positions_read:
                        image_positions_read.append(image_position)
        if load_images and image_positions_read:
            await self.read_images(user_uuid, workspace, image_positions_read)
        return SourceReadResult(
            passages=passages,
            block_positions=block_positions_read,
            image_positions=image_positions_read,
        )

    async def read_images(
        self,
        user_uuid: str,
        workspace: SourceWorkspace,
        image_positions: list[int],
    ) -> list[int]:
        if not image_positions:
            return []
        entries = [
            workspace.image_registry[position] for position in image_positions
        ]
        images = await read_images(self._source_evidence, user_uuid, entries)
        for position, image in zip(image_positions, images, strict=True):
            workspace.image_registry[position].data = image.data
        return image_positions.copy()
