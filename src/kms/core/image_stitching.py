"""Assemble image regions at their original relative positions."""

from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

from PIL import Image

from kms.core.media_budget import ImagePayloadBudget
from kms.core.tokenization.budget import InputBudgetExceeded


def stitch_positioned_crops(
    crops: list[tuple[Path, tuple[int, int, int, int], int]],
    *,
    budget: ImagePayloadBudget,
) -> tuple[bytes, list[tuple[int, int, int, int]]]:
    """Return positioned page crops and their bounds in the resulting PNG."""
    left = min(box[0] for _, box, _ in crops)
    top = min(box[1] + offset for _, box, offset in crops)
    right = max(box[2] for _, box, _ in crops)
    bottom = max(box[3] + offset for _, box, offset in crops)
    width, height = right - left, bottom - top
    if width * height > budget.max_pixels:
        raise InputBudgetExceeded(
            'Active image evidence exceeds the configured payload budget'
        )
    bounds = [
        (
            box[0] - left,
            box[1] + offset - top,
            box[2] - left,
            box[3] + offset - top,
        )
        for _, box, offset in crops
    ]
    with ExitStack() as stack:
        images = {}
        for path, _, _ in crops:
            if path not in images:
                images[path] = stack.enter_context(Image.open(path))
        output = BytesIO()
        with Image.new('RGB', (width, height), 'white') as stitched:
            for (path, box, _), destination in zip(crops, bounds, strict=True):
                with images[path].crop(box) as image:
                    stitched.paste(image, destination[:2])
            stitched.save(output, format='PNG')
    payload = output.getvalue()
    if len(payload) > budget.max_bytes:
        raise InputBudgetExceeded(
            'Active image evidence exceeds the configured payload budget'
        )
    return payload, bounds
