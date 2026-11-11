"""Load, encode, and prepare native images from immutable assets."""

import base64
from collections.abc import Sequence
from functools import partial
from io import BytesIO
from pathlib import Path

from PIL import Image

from kms.artifacts import ContentAddressedArtifactStore
from kms.concurrency import run_blocking_work
from kms.media.budget import ImagePayloadBudget, check_image_payload
from kms.media.models import ImageAsset, Material


async def load_content_images(
    images: Sequence[ImageAsset], *, budget: ImagePayloadBudget | None
) -> list[bytes]:
    """Read assets off-thread and enforce limits for their unique keys."""
    return await run_blocking_work(
        partial(load_content_images_sync, images, budget=budget)
    )


def load_content_images_sync(
    images: Sequence[ImageAsset], *, budget: ImagePayloadBudget | None
) -> list[bytes]:
    """Read assets and enforce aggregate limits, deduplicating I/O by key."""
    payloads_by_key: dict[tuple[str, ...], bytes] = {}
    for image in images:
        if image.key not in payloads_by_key:
            payloads_by_key[image.key] = _read_image(image)
    unique_payloads = list(payloads_by_key.values())
    if budget is not None:
        check_image_payload(unique_payloads, budget)
    return [payloads_by_key[image.key] for image in images]


async def encode_content_images(images: Sequence[ImageAsset]) -> list[str]:
    """Encode supplied image occurrences as same-order data URIs."""
    return await run_blocking_work(encode_content_images_sync, images)


def encode_content_images_sync(images: Sequence[ImageAsset]) -> list[str]:
    """Encode supplied native image occurrences as same-order data URIs."""
    payloads = load_content_images_sync(images, budget=None)
    return _encode_image_data_uris(payloads)


async def prepare_image_uris(
    materials: Sequence[Material],
) -> dict[tuple[str, ...], str]:
    """Prepare one data URI per unique image key referenced by materials."""
    images_by_key: dict[tuple[str, ...], ImageAsset] = {}
    for material in materials:
        for image in material.images:
            images_by_key.setdefault(image.key, image)
    images = list(images_by_key.values())
    uris = await encode_content_images(images)
    return dict(zip(images_by_key, uris, strict=True))


def _encode_image_data_uris(payloads: list[bytes]) -> list[str]:
    data_uris = []
    for payload in payloads:
        with Image.open(BytesIO(payload)) as opened:
            mime_type = Image.MIME[opened.format]
        encoded = base64.b64encode(payload).decode('ascii')
        data_uris.append(f'data:{mime_type};base64,{encoded}')
    return data_uris


def _read_image(image: ImageAsset) -> bytes:
    return (
        ContentAddressedArtifactStore.read(image.payload)
        if isinstance(image.payload, Path)
        else image.payload
    )
