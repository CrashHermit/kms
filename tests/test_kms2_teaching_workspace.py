"""Session workspace persistence and native-image rehydration."""

import asyncio

import pytest

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import StagedSourceSnapshot
from kms2.core.model.teaching.workspace import (
    ImageInspection,
    TeachingEvidenceWorkspace,
    TeachingImageRegistryEntry,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.service.teaching_workspace import (
    restore_workspace,
    serialize_workspace,
)


def test_workspace_restores_inspected_image_and_active_evidence(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'original-image-bytes')
    source = Source(key='source')
    asset = VisualAsset(path=str(image_path))
    block = SourceBlock(block_type='image', content='caption', assets=[asset])
    staged = StagedSourceSnapshot(
        source=source, pages=[SourcePage(index=7, blocks=[block])]
    )
    workspace = TeachingEvidenceWorkspace(
        staged_sources=[staged],
        acquired_sources=[staged],
        image_registry=[
            TeachingImageRegistryEntry(
                asset=asset, source=source, data=b'original-image-bytes'
            )
        ],
        inspected_image_positions={0},
        active_image_positions=[0],
        receipts=[
            ImageInspection(
                source=source,
                asset=asset,
                block_uuids=[block.uuid],
                data=b'original-image-bytes',
            )
        ],
    )

    stored = serialize_workspace(workspace)
    assert 'original-image-bytes' not in stored
    restored = asyncio.run(restore_workspace(stored))
    assert restored == workspace
    assert restored.active_image_positions == [0]
    assert restored.staged_sources[0].pages[0].index == 7


def test_restore_fails_when_an_inspected_image_is_missing(tmp_path):
    asset = VisualAsset(path=str(tmp_path / 'missing.png'))
    source = Source(key='source')
    workspace = TeachingEvidenceWorkspace(
        image_registry=[
            TeachingImageRegistryEntry(
                asset=asset, source=source, data=b'bytes'
            )
        ],
        inspected_image_positions={0},
        receipts=[
            ImageInspection(
                source=source, asset=asset, block_uuids=[], data=b'bytes'
            )
        ],
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(restore_workspace(serialize_workspace(workspace)))
