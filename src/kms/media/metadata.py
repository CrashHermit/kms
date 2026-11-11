"""Native-image metadata shared by token estimators and payload limits."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from kms.media.models import ImageAsset


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Image dimensions and encoded payload size."""

    width: int
    height: int
    byte_count: int


class ImageMetadataCache:
    """Cache image headers and encoded sizes for one owner lifetime."""

    def __init__(self) -> None:
        self._metadata: dict[tuple[str, ...], ImageMetadata] = {}

    def measure(self, image: ImageAsset) -> ImageMetadata:
        """Read cached dimensions and encoded size for one native image."""
        metadata = self._metadata.get(image.key)
        if metadata is not None:
            return metadata
        if isinstance(image.payload, Path):
            byte_count = image.payload.stat().st_size
            with Image.open(image.payload) as opened:
                width, height = opened.size
        else:
            byte_count = len(image.payload)
            with Image.open(BytesIO(image.payload)) as opened:
                width, height = opened.size
        metadata = ImageMetadata(width, height, byte_count)
        self._metadata[image.key] = metadata
        return metadata
