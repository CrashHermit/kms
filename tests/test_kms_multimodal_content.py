import asyncio
import base64
from io import BytesIO

import dspy
import pytest
from PIL import Image

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.media.content import collect_image_assets
from kms.media.images import (
    encode_content_images,
    encode_content_images_sync,
    load_content_images,
    load_content_images_sync,
    prepare_image_uris,
)
from kms.media.models import ImageAsset, ImagePart, Material, TextPart


def _png(color: str) -> bytes:
    stream = BytesIO()
    Image.new('RGB', (3, 2), color).save(stream, format='PNG')
    return stream.getvalue()


class _OneTokenCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1] * len(texts)


def _sized_png(size: tuple[int, int]) -> bytes:
    stream = BytesIO()
    Image.new('RGB', size, 'white').save(stream, format='PNG')
    return stream.getvalue()


def test_batch_packing_counts_image_tokens_and_keeps_media_limits():
    async def run() -> None:
        first = ImageAsset(('image', 'first'), _sized_png((64, 64)))
        second = ImageAsset(('image', 'second'), _sized_png((64, 64)))
        repeated = ImageAsset(('image', 'first'), first.payload)
        budget = TokenBudget(MaterialTokenEstimator(_OneTokenCounter()), 6)
        media_budget = ImagePayloadBudget(4, 100_000, 100_000)

        def project(items):
            return {
                'items': [
                    Material(
                        parts=(
                            TextPart(text='caption'),
                            ImagePart(asset=item),
                        )
                    )
                    for item in items
                ]
            }

        batches = await pack_content_batches(
            [[first, second]],
            project=project,
            budget=budget,
            image_budget=media_budget,
        )
        assert batches == [[first], [second]]
        assert await pack_content_batches(
            [[first]],
            project=project,
            budget=budget,
            image_budget=media_budget,
        ) == [[first]]
        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [[first]],
                project=project,
                budget=TokenBudget(
                    MaterialTokenEstimator(_OneTokenCounter()), 5
                ),
                image_budget=media_budget,
            )
        assert await pack_content_batches(
            [[first, repeated]],
            project=project,
            budget=budget,
            image_budget=media_budget,
        ) == [[first, repeated]]
        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [[first]],
                project=project,
                budget=TokenBudget(
                    MaterialTokenEstimator(_OneTokenCounter()), 1000
                ),
                image_budget=ImagePayloadBudget(0, 100_000, 100_000),
            )

    asyncio.run(run())


def test_material_preserves_order_and_duplicate_occurrences():
    red = ImageAsset(('path', 'red'), _png('red'))
    blue = ImageAsset(('research-pdf-page', '1', '0'), _png('blue'))
    material = Material(
        parts=(
            TextPart(text='caption'),
            ImagePart(asset=red),
            TextPart(text='between'),
            ImagePart(asset=blue),
            ImagePart(asset=red),
        )
    )
    image_only = Material.from_image(blue)
    assert [part.type for part in material.parts] == [
        'text',
        'image',
        'text',
        'image',
        'image',
    ]
    assert material.text == 'captionbetween'
    assert material.images == (red, blue, red)
    assert image_only.text is None
    assert image_only.images == (blue,)
    assert Material.from_text(None).parts == ()
    assert Material.from_text('').text == ''


def test_asset_collection_deduplicates_keys_in_encounter_order():
    from pydantic import BaseModel

    class Nested(BaseModel):
        items: list[Material]

    red = ImageAsset(('image', 'red'), _png('red'))
    blue = ImageAsset(('image', 'blue'), _png('blue'))
    red_duplicate = ImageAsset(('image', 'red'), b'different payload ignored')
    value = Nested(
        items=[
            Material(parts=(ImagePart(asset=red), ImagePart(asset=blue))),
            Material.from_image(red_duplicate),
        ]
    )
    assert collect_image_assets({'nested': (value, [blue])}) == [red, blue]


def test_image_loading_and_encoding_preserve_occurrences_and_deduplicate_keys(
    tmp_path, monkeypatch
):
    image_path = tmp_path / 'red.png'
    image_path.write_bytes(_png('red'))
    red = ImageAsset(('path', str(image_path)), image_path)
    duplicate = ImageAsset(('path', str(image_path)), image_path)
    images = [red, duplicate]

    from kms.artifacts import ContentAddressedArtifactStore

    original_read = ContentAddressedArtifactStore.read
    reads = []

    def read_once(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(ContentAddressedArtifactStore, 'read', read_once)
    assert load_content_images_sync(images, budget=None) == [_png('red')] * 2
    assert len(reads) == 1
    assert load_content_images_sync([], budget=None) == []
    assert len(encode_content_images_sync([])) == 0
    assert len(encode_content_images_sync(images)) == 2
    assert len(reads) == 2

    materials = [
        Material(
            parts=(
                TextPart(text='before'),
                ImagePart(asset=duplicate),
                TextPart(text='after'),
                ImagePart(asset=red),
            )
        )
    ]
    image_uris = asyncio.run(prepare_image_uris(materials))
    assert list(image_uris) == [red.key]
    assert base64.b64decode(
        image_uris[red.key].split(',', maxsplit=1)[1]
    ) == _png('red')


def test_async_image_loading_enforces_aggregate_budget(tmp_path):
    red = tmp_path / 'red.png'
    red.write_bytes(_png('red'))
    entries = [
        ImageAsset(('path', str(red)), red),
        ImageAsset(('image', 'blue'), _png('blue')),
    ]
    limit = ImagePayloadBudget(max_images=1, max_bytes=10000, max_pixels=6)

    async def run() -> None:
        with pytest.raises(InputBudgetExceeded):
            await load_content_images(entries, budget=limit)
        assert await load_content_images([], budget=limit) == []
        encoded = await encode_content_images(entries)
        assert len(encoded) == len(entries)
        assert all(uri.startswith('data:image/png;base64,') for uri in encoded)

    asyncio.run(run())


def test_real_adapter_emits_native_image_without_backend_identity():
    class ReadSignature(dspy.Signature):
        material: Material = dspy.InputField()
        content_images: list[dspy.Image] = dspy.InputField()
        answer: str = dspy.OutputField()

    asset = ImageAsset(('path', '/private/asset.png'), _png('red'))
    material = Material.from_text('source caption')
    payloads = load_content_images_sync([asset], budget=None)
    messages = dspy.ChatAdapter().format(
        ReadSignature,
        [],
        {'material': material, 'content_images': [dspy.Image(payloads[0])]},
    )
    parts = [
        part
        for message in messages
        if isinstance(message['content'], list)
        for part in message['content']
    ]
    assert any(part.get('type') == 'image_url' for part in parts)
    assert '/private/asset.png' not in str(messages)
    assert 'source caption' in str(messages)
