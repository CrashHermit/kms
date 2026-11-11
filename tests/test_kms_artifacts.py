"""Tests for durable content-addressed source artifacts."""

import asyncio
import hashlib

import pytest

from kms.artifacts import (
    ArtifactIntegrityError,
    ContentAddressedArtifactStore,
)
from kms.media.images import load_content_images
from kms.media.models import ImageAsset


def test_artifact_store_deduplicates_and_rejects_corrupted_content(
    tmp_path,
) -> None:
    data = b'captured source image'
    digest = hashlib.sha256(data).hexdigest()
    store = ContentAddressedArtifactStore(tmp_path)

    first = store.write(data, suffix='.png')
    duplicate = store.write(data, suffix='.png')

    assert first.is_absolute()
    assert first == duplicate
    assert (
        first.relative_to(tmp_path).as_posix() == f'{digest[:2]}/{digest}.png'
    )
    assert store.read(first) == data

    first.write_bytes(b'corrupted')
    with pytest.raises(ArtifactIntegrityError):
        store.read(first)
    with pytest.raises(ArtifactIntegrityError):
        store.write(data, suffix='.png')
    with pytest.raises(ArtifactIntegrityError):
        asyncio.run(
            load_content_images(
                [ImageAsset(('path', str(first)), first)],
                budget=None,
            )
        )
