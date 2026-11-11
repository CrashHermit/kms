"""Packaged images used by source-pedagogical demonstrations."""

from pathlib import Path

from kms.media.models import ImageAsset

ASSETS_DIRECTORY = Path(__file__).with_name('assets')


def source_image(name: str) -> ImageAsset:
    """Reference a checked-in source-processing image asset."""
    return ImageAsset(
        key=('source-example', name),
        payload=ASSETS_DIRECTORY / name,
    )
