"""Operational limits for aggregate native-image model payloads."""

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from kms2.core.windowing import InputBudgetExceeded


@dataclass(frozen=True, slots=True)
class ImagePayloadBudget:
    """Independent aggregate limits for native image evidence payloads."""

    max_images: int
    max_bytes: int
    max_pixels: int


def check_image_payload(
    images: list[bytes], budget: ImagePayloadBudget
) -> None:
    """Reject an image batch that exceeds any configured aggregate limit."""
    total_bytes = sum(len(image) for image in images)
    total_pixels = 0
    for payload in images:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            total_pixels += width * height
    if (
        len(images) > budget.max_images
        or total_bytes > budget.max_bytes
        or total_pixels > budget.max_pixels
    ):
        raise InputBudgetExceeded(
            'Active image evidence exceeds the configured payload budget'
        )
