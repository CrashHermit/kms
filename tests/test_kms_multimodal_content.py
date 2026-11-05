import asyncio
import base64
from io import BytesIO

import dspy
import pytest
from PIL import Image

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.content import MultimodalContent
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.service.content import (
    encode_content_images,
    load_content_images,
    pack_content_batches,
)


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
        first = ContentImage(('image', 'first'), _sized_png((64, 64)))
        second = ContentImage(('image', 'second'), _sized_png((64, 64)))
        repeated = ContentImage(('image', 'first'), first.payload)
        budget = TokenBudget(MaterialTokenEstimator(_OneTokenCounter()), 6)
        media_budget = ImagePayloadBudget(4, 100_000, 100_000)

        def project(items, builder):
            return {
                'items': [
                    builder.build(text='caption', images=[item])
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

        exact_boundary = await pack_content_batches(
            [[first]],
            project=project,
            budget=budget,
            image_budget=media_budget,
        )
        assert exact_boundary == [[first]]

        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [[first]],
                project=project,
                budget=TokenBudget(
                    MaterialTokenEstimator(_OneTokenCounter()), 5
                ),
                image_budget=media_budget,
            )

        deduplicated = await pack_content_batches(
            [[first, repeated]],
            project=project,
            budget=budget,
            image_budget=media_budget,
        )
        assert deduplicated == [[first, repeated]]

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


def test_material_preserves_roles_and_resets_request_ordinals():
    red = ContentImage(('path', 'red'), _png('red'))
    blue = ContentImage(('research-pdf-page', '1', '0'), _png('blue'))
    builder = MultimodalContentBuilder()
    assert builder.build(text='plain').model_dump() == {
        'text': 'plain',
        'image_positions': [],
    }
    mixed = builder.build(text='caption', images=[red, blue, red])
    visual = builder.build(text=None, images=[blue])
    assert mixed.text == 'caption'
    assert mixed.image_positions == [0, 1]
    assert visual.text is None
    assert visual.image_positions == [1]
    assert builder.images == [red, blue]
    assert MultimodalContentBuilder().build(
        text=None, images=[blue]
    ).image_positions == [0]


def test_whole_item_packing_and_actual_image_loading(tmp_path):
    async def run():
        red = tmp_path / 'red.png'
        red.write_bytes(_png('red'))
        entries = [
            ContentImage(('path', str(red)), red),
            ContentImage(('research-image', '2', 'blue'), _png('blue')),
        ]
        limit = ImagePayloadBudget(max_images=1, max_bytes=10000, max_pixels=6)

        def project(items, builder):
            return {
                'items': [
                    builder.build(text='caption', images=[item])
                    for item in items
                ]
            }

        batches = await pack_content_batches(
            [entries], project=project, budget=None, image_budget=limit
        )
        assert batches == [[entries[0]], [entries[1]]]
        for entry, expected in zip(
            entries, [(255, 0, 0), (0, 0, 255)], strict=True
        ):
            payloads = await load_content_images([entry], budget=limit)
            with Image.open(BytesIO(payloads[0])) as image:
                assert image.getpixel((0, 0)) == expected
        with pytest.raises(InputBudgetExceeded):
            await load_content_images(entries, budget=limit)
        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [entries],
                project=project,
                budget=None,
                image_budget=ImagePayloadBudget(0, 10000, 10000),
            )
        missing = ContentImage(('path', 'missing'), tmp_path / 'missing.png')
        assert (
            await pack_content_batches(
                [[missing]],
                project=project,
                budget=None,
                image_budget=limit,
                is_eligible=lambda items: len(items) > 1,
            )
            == []
        )
        assert await load_content_images([], budget=limit) == []
        assert await encode_content_images([]) == []
        encoded = await encode_content_images(entries)
        assert len(encoded) == len(entries)
        assert all(uri.startswith('data:image/png;base64,') for uri in encoded)
        assert [
            base64.b64decode(uri.split(',', maxsplit=1)[1]) for uri in encoded
        ] == [_png('red'), _png('blue')]

    asyncio.run(run())


def test_real_adapter_emits_native_image_without_backend_identity():
    class ReadSignature(dspy.Signature):
        material: MultimodalContent = dspy.InputField()
        content_images: list[dspy.Image] = dspy.InputField()
        answer: str = dspy.OutputField()

    builder = MultimodalContentBuilder()
    material = builder.build(
        text='source caption',
        images=[ContentImage(('path', '/private/asset.png'), _png('red'))],
    )
    payloads = asyncio.run(load_content_images(builder.images, budget=None))
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
