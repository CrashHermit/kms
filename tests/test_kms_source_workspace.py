"""Behavioral checks for graph-backed source workspaces."""

import asyncio

import pytest

from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceImage,
    SourcePassage,
)
from kms.core.model.source_processing.admission import SourceSnapshot
from kms.core.model.source_workspace import (
    ResearchWorkspace,
    SourceWorkspace,
    SourceWorkspaceBlockEntry,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.source_workspace import register_source
from kms.service.source_workspace import SourceWorkspaceService


def _passage(source: Source, block: SourceBlock) -> SourcePassage:
    return SourcePassage(
        source=source,
        selected_block_uuids=[block.uuid],
        blocks=[SourceEvidenceBlock(block=block, page_indexes=[0])],
        groups=[],
    )


class _Evidence:
    def __init__(self, passages=None, images=None) -> None:
        self.passages = passages or []
        self.images = images or []
        self.reads: list[tuple[list[str], dict[str, object]]] = []
        self.image_references: list[tuple[str, int, str]] = []

    async def read_passages(self, user_uuid, block_uuids, **options):
        self.reads.append((block_uuids, options))
        return self.passages

    async def read_images_at_revisions(self, user_uuid, references):
        self.image_references = references
        return self.images


def test_read_blocks_uses_owned_graph_content_not_cached_text():
    source = Source(uuid='source', key='Source', revision=2)
    cached = SourceBlock(uuid='block', block_type='paragraph', content='cached')
    canonical = cached.model_copy(update={'content': 'graph content'})
    workspace = SourceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[0], block=cached
            )
        ]
    )
    evidence = _Evidence(passages=[_passage(source, canonical)])

    result = asyncio.run(
        SourceWorkspaceService(evidence).read_blocks('owner', workspace, [0])
    )

    assert result.passages[0].blocks[0].block.content == 'graph content'
    assert evidence.reads == [
        (['block'], {'expand_memberships': False, 'before': 0, 'after': 0})
    ]
    assert workspace.block_registry[0].block.content == 'graph content'


@pytest.mark.parametrize(
    'returned_passages',
    [
        [],
        [
            _passage(
                Source(uuid='source', key='Source', revision=3),
                SourceBlock(
                    uuid='block',
                    block_type='paragraph',
                    content='wrong revision',
                ),
            )
        ],
    ],
)
def test_read_blocks_rejects_missing_or_wrong_revision(returned_passages):
    source = Source(uuid='source', key='Source', revision=2)
    block = SourceBlock(uuid='block', block_type='paragraph', content='cached')
    workspace = SourceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[0], block=block
            )
        ]
    )
    evidence = _Evidence(passages=returned_passages)

    with pytest.raises(
        ValueError, match='Requested source revision is no longer available'
    ):
        asyncio.run(
            SourceWorkspaceService(evidence).read_blocks(
                'owner', workspace, [0]
            )
        )


def test_register_source_keeps_only_summary_and_counts_asset_occurrences():
    source = Source(uuid='source', key='Source')
    first_asset = VisualAsset(
        uuid='asset-1', path='first.png', content_sha256='same-bytes'
    )
    second_asset = VisualAsset(
        uuid='asset-2', path='second.png', content_sha256='same-bytes'
    )
    first_block = SourceBlock(
        uuid='block', block_type='image', content='figure', assets=[first_asset]
    )
    second_block = SourceBlock(
        uuid='block',
        block_type='image',
        content='figure',
        assets=[second_asset],
    )
    workspace = ResearchWorkspace()

    position = register_source(
        workspace,
        SourceSnapshot(
            source=source,
            pages=[
                SourcePage(index=0, blocks=[first_block]),
                SourcePage(index=1, blocks=[second_block]),
            ],
        ),
    )

    entry = workspace.acquired_sources[position]
    assert (entry.page_count, entry.block_count, entry.image_count) == (2, 1, 2)
    assert not hasattr(entry, 'pages')
    assert workspace.block_registry[0].page_indexes == [0, 1]
    assert [item.asset.uuid for item in workspace.image_registry] == [
        'asset-1',
        'asset-2',
    ]


def test_read_blocks_loads_image_bytes_from_the_owned_revision():
    source = Source(uuid='source', key='Source', revision=4)
    asset = VisualAsset(
        uuid='asset', path='old-path.png', content_sha256='digest'
    )
    block = SourceBlock(
        uuid='block', block_type='image', content='diagram', assets=[asset]
    )
    workspace = SourceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[0], block=block
            )
        ]
    )
    evidence = _Evidence(
        passages=[_passage(source, block)],
        images=[SourceImage(asset=asset, data=b'owned bytes')],
    )

    result = asyncio.run(
        SourceWorkspaceService(evidence).read_blocks('owner', workspace, [0])
    )

    assert result.image_positions == [0]
    assert evidence.image_references == [('source', 4, 'asset')]
    assert workspace.image_registry[0].data == b'owned bytes'
