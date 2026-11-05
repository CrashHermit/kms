"""Load and pack bounded native-image model inputs."""

import asyncio
import base64
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import TypeAdapter

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget, check_image_payload
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.images import ImageMetadataCache
from kms.service.artifacts import ContentAddressedArtifactStore


async def load_content_images(
    images: list[ContentImage], *, budget: ImagePayloadBudget | None
) -> list[bytes]:
    """Read and enforce aggregate limits off-thread for one image catalog."""
    if not images:
        return []
    return await asyncio.to_thread(_load_images, images, budget)


async def encode_content_images(images: list[ContentImage]) -> list[str]:
    """Encode the supplied native image catalog as same-order data URIs."""
    payloads = await load_content_images(images, budget=None)
    if not payloads:
        return []
    return await asyncio.to_thread(_encode_image_data_uris, payloads)


def _encode_image_data_uris(payloads: list[bytes]) -> list[str]:
    data_uris = []
    for payload in payloads:
        with Image.open(BytesIO(payload)) as opened:
            mime_type = Image.MIME[opened.format]
        encoded = base64.b64encode(payload).decode('ascii')
        data_uris.append(f'data:{mime_type};base64,{encoded}')
    return data_uris


def _load_images(
    images: list[ContentImage], budget: ImagePayloadBudget | None
) -> list[bytes]:
    payloads = [
        ContentAddressedArtifactStore.read(image.payload)
        if isinstance(image.payload, Path)
        else image.payload
        for image in images
    ]
    if budget is not None:
        check_image_payload(payloads, budget)
    return payloads


def _image_metadata_fits(
    images: list[ContentImage],
    budget: ImagePayloadBudget,
    image_metadata: ImageMetadataCache,
) -> bool:
    if len(images) > budget.max_images:
        return False
    total_bytes = 0
    total_pixels = 0
    for entry in images:
        metadata = image_metadata.measure(entry)
        total_bytes += metadata.byte_count
        total_pixels += metadata.width * metadata.height
    return total_bytes <= budget.max_bytes and total_pixels <= budget.max_pixels


def _content_fits(
    inputs: dict[str, object],
    images: list[ContentImage],
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget,
    image_metadata: ImageMetadataCache,
) -> tuple[bool, bool]:
    if budget is not None:
        token_count = budget.counter.estimate(
            text=_serialized_inputs(inputs),
            images=images,
        )
        if token_count > budget.token_limit:
            return False, True
    return True, _image_metadata_fits(images, image_budget, image_metadata)


def _serialized_inputs(inputs: dict[str, object]) -> str:
    return (
        TypeAdapter(dict[str, object])
        .dump_json(
            {**inputs, 'content_images': []},
            serialize_as_any=True,
        )
        .decode('utf-8')
    )


async def pack_content_batches[Item](
    batches: list[list[Item]],
    *,
    project: Callable[
        [list[Item], MultimodalContentBuilder], dict[str, object]
    ],
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget,
    is_eligible: Callable[[list[Item]], bool] | None = None,
) -> list[list[Item]]:
    """Refine ordered whole-item batches by serialized text and image capacity."""
    eligible = is_eligible or (lambda items: bool(items))

    packed: list[list[Item]] = []
    image_metadata = (
        budget.counter.image_metadata
        if budget is not None
        else ImageMetadataCache()
    )

    async def fits(items: list[Item]) -> tuple[bool, bool]:
        builder = MultimodalContentBuilder()
        inputs = project(items, builder)
        return await asyncio.to_thread(
            _content_fits,
            inputs,
            builder.images,
            budget,
            image_budget,
            image_metadata,
        )

    for incoming in batches:
        current: list[Item] = []
        for item in incoming:
            candidate = [*current, item]
            if not eligible(candidate):
                current = candidate
                continue
            text_fits, image_fits = await fits(candidate)
            if text_fits and image_fits:
                current = candidate
                continue
            if current and eligible(current):
                packed.append(current)
            current = [item]
            if eligible(current):
                text_fits, image_fits = await fits(current)
                if not text_fits:
                    raise InputBudgetExceeded(
                        'Content exceeds the configured input budget'
                    )
                if not image_fits:
                    raise InputBudgetExceeded(
                        'Image content exceeds the configured payload budget'
                    )
        if current and eligible(current):
            packed.append(current)
    return packed
