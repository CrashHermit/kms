"""Canonical source-block image composition."""

import hashlib
import os
import tempfile
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

from PIL import Image

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.media.budget import ImagePayloadBudget
from kms.source.models.source import SourceBlock


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


def _fuse_image_bytes(
    parts: list[tuple[int, SourceBlock]],
    page_images: dict[int, str],
    budget: ImagePayloadBudget,
) -> Path:
    page_offsets: dict[int, int] = {}
    offset = 0
    for page_index, _ in parts:
        page_path = Path(page_images[page_index])
        page_offsets[page_index] = offset
        with Image.open(page_path) as page_image:
            offset += page_image.height

    crops = []
    for page_index, block in parts:
        if block.crop_bbox is None:
            raise ValueError('Image block is missing original crop geometry')
        crops.append(
            (
                Path(page_images[page_index]),
                block.crop_bbox,
                page_offsets[page_index],
            )
        )
    payload, _ = stitch_positioned_crops(crops, budget=budget)
    digest = hashlib.sha256(payload).hexdigest()
    crop_path = parts[0][1].crop_path
    if crop_path is None:
        raise ValueError('Image block is missing its original crop path')
    parent = Path(crop_path).parent
    target = parent / f'Fused_{digest}.png'
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


async def fuse_image_blocks(
    parts: list[tuple[int, SourceBlock]],
    *,
    page_images: dict[int, str],
    budget: ImagePayloadBudget,
) -> SourceBlock:
    """Fuse original page crops once and return the first block with one image."""
    if len(parts) < 2:
        raise ValueError('Image fusion requires at least two source blocks')
    image_path = await run_blocking_work(
        _fuse_image_bytes, parts, page_images, budget
    )
    first = parts[0][1]
    return first.model_copy(
        update={
            'image_path': str(image_path),
            'image_sha256': None,
            'image_page_indexes': [page_index for page_index, _ in parts],
        }
    )
