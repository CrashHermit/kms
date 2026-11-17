"""Packaged images shared by learning-component demonstrations."""

from pathlib import Path

from kms.media.models import ImageAsset

ASSETS_DIRECTORY = Path(__file__).with_name('assets')


def learning_image(name: str) -> ImageAsset:
    """Reference a checked-in learning demonstration image."""
    return ImageAsset(
        key=('learning-example', name),
        payload=ASSETS_DIRECTORY / name,
    )
