import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.media.content import collect_image_assets
from kms.media.images import load_content_images
from kms.media.models import ImageAsset
from kms.source.context import (
    SourceBlockWindow,
    count_source_tokens,
    project_source_material,
    project_source_window,
    select_window,
)
from kms.source.models.source import BlockType, SourceBlock


def _png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new('RGB', (8, 8), color).save(path, format='PNG')


def _block(
    content: str | None,
    image_path: str | None = None,
    block_type: BlockType = BlockType.PARAGRAPH,
) -> SourceBlock:
    return SourceBlock(
        block_type=block_type if image_path is None else BlockType.IMAGE,
        content=content,
        image_path=image_path,
        image_page_indexes=[0] if image_path is not None else [],
    )


def _pixel(payload: bytes) -> tuple[int, int, int]:
    with Image.open(BytesIO(payload)) as image:
        return image.getpixel((0, 0))


class _LengthCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def test_source_projection_preserves_block_order_and_image_provenance(tmp_path):
    red_path = tmp_path / 'red.png'
    blue_path = tmp_path / 'blue.png'
    _png(red_path, (255, 0, 0))
    _png(blue_path, (0, 0, 255))
    window = SourceBlockWindow(
        context_before=[_block('before', str(red_path))],
        target=[_block('figure', str(blue_path))],
        context_after=[_block('after', str(red_path))],
    )

    projected = project_source_window(window)
    before = projected.context_before[0].material
    target = projected.target[0].material
    after = projected.context_after[0].material

    assert [part.type for part in before.parts] == ['text', 'image']
    assert before.parts[0].text == 'before'
    assert before.images[0].key == ('path', str(red_path))
    assert target.images[0].key == ('path', str(blue_path))
    assert after.images[0].key == ('path', str(red_path))
    assert _pixel(_read(str(before.images[0].payload))) == (255, 0, 0)


def test_source_material_keeps_evidence_image_occurrences_in_order(tmp_path):
    red_path = tmp_path / 'red.png'
    blue_path = tmp_path / 'blue.png'
    _png(red_path, (255, 0, 0))
    _png(blue_path, (0, 0, 255))
    blocks = [
        _block('first', str(red_path)),
        SourceBlock(
            block_type=BlockType.PARAGRAPH,
            content='crop-only text',
            crop_path=str(tmp_path / 'correction-crop.png'),
        ),
        _block('second', str(blue_path)),
        _block('duplicate path', str(red_path)),
    ]

    material = project_source_material('exact annotation text', blocks)

    assert material.text == 'exact annotation text'
    assert material.images == (
        ImageAsset(('path', str(red_path)), red_path),
        ImageAsset(('path', str(blue_path)), blue_path),
        ImageAsset(('path', str(red_path)), red_path),
    )
    assert collect_image_assets(material) == list(material.images[:2])


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def test_source_window_costs_include_direct_images_but_not_crop_paths(tmp_path):
    image_path = tmp_path / 'figure.png'
    _png(image_path, (255, 0, 0))
    blocks = [
        SourceBlock(
            uuid='text',
            block_type=BlockType.PARAGRAPH,
            content='x',
            crop_path=str(tmp_path / 'missing-text-crop.png'),
        ),
        SourceBlock(
            uuid='image',
            block_type=BlockType.IMAGE,
            content=None,
            crop_path=str(tmp_path / 'missing-image-crop.png'),
            image_path=str(image_path),
            image_page_indexes=[0],
        ),
        SourceBlock(
            uuid='target',
            block_type=BlockType.PARAGRAPH,
            content='',
        ),
    ]

    costs = count_source_tokens(
        blocks, (MaterialTokenEstimator(_LengthCounter()),)
    )

    assert costs == [1, 2, 0]
    assert count_source_tokens(
        [
            SourceBlock(
                uuid='mixed',
                block_type=BlockType.IMAGE,
                content='x',
                crop_path=str(tmp_path / 'missing-mixed-crop.png'),
                image_path=str(image_path),
                image_page_indexes=[0],
            )
        ],
        (MaterialTokenEstimator(_LengthCounter()),),
    ) == [3]
    assert select_window(
        blocks,
        [2],
        token_counts=costs,
        backward_budget=2,
    ).context_before == [blocks[1]]
    assert (
        select_window(
            blocks,
            [2],
            token_counts=costs,
            backward_budget=3,
        ).context_before
        == blocks[:2]
    )


def test_image_limit_splits_items_and_rebuilds_each_request_catalog(tmp_path):
    async def run() -> None:
        red_path = tmp_path / 'red.png'
        blue_path = tmp_path / 'blue.png'
        _png(red_path, (255, 0, 0))
        _png(blue_path, (0, 0, 255))
        windows = [
            SourceBlockWindow(target=[_block('red paragraph', str(red_path))]),
            SourceBlockWindow(
                target=[_block('blue paragraph', str(blue_path))]
            ),
        ]

        def project(items: list[SourceBlockWindow]) -> dict[str, object]:
            return {'evidence': [project_source_window(item) for item in items]}

        batches = await pack_content_batches(
            [windows],
            project=project,
            budget=None,
            image_budget=ImagePayloadBudget(
                max_images=1,
                max_bytes=100_000,
                max_pixels=10_000,
            ),
        )

        assert batches == [[windows[0]], [windows[1]]]
        colors = []
        for batch in batches:
            evidence = project(batch)['evidence']
            images = await load_content_images(
                collect_image_assets(evidence),
                budget=ImagePayloadBudget(
                    max_images=1,
                    max_bytes=100_000,
                    max_pixels=10_000,
                ),
            )
            colors.append(_pixel(images[0]))
        assert colors == [(255, 0, 0), (0, 0, 255)]

    asyncio.run(run())


def test_aggregate_byte_and_pixel_limits_split_visual_items(tmp_path):
    async def run() -> None:
        paths = [tmp_path / 'red.png', tmp_path / 'blue.png']
        _png(paths[0], (255, 0, 0))
        _png(paths[1], (0, 0, 255))
        payloads = [_read(str(path)) for path in paths]
        windows = [
            SourceBlockWindow(target=[_block(path.stem, str(path))])
            for path in paths
        ]

        def project(items: list[SourceBlockWindow]) -> dict[str, object]:
            return {'evidence': [project_source_window(item) for item in items]}

        budgets = [
            ImagePayloadBudget(
                max_images=2,
                max_bytes=max(map(len, payloads)),
                max_pixels=128,
            ),
            ImagePayloadBudget(
                max_images=2,
                max_bytes=sum(map(len, payloads)),
                max_pixels=64,
            ),
        ]
        for image_budget in budgets:
            batches = await pack_content_batches(
                [windows],
                project=project,
                budget=None,
                image_budget=image_budget,
            )
            assert batches == [[windows[0]], [windows[1]]]

    asyncio.run(run())


def test_image_capacity_errors_are_reported_before_prediction(tmp_path):
    async def run() -> None:
        image_path = tmp_path / 'image.png'
        _png(image_path, (0, 255, 0))
        window = SourceBlockWindow(target=[_block('green', str(image_path))])

        def project(items: list[SourceBlockWindow]) -> dict[str, object]:
            return {'evidence': [project_source_window(item) for item in items]}

        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [[window]],
                project=project,
                budget=None,
                image_budget=ImagePayloadBudget(
                    max_images=1,
                    max_bytes=100_000,
                    max_pixels=63,
                ),
            )

        with pytest.raises(InputBudgetExceeded):
            await load_content_images(
                [ImageAsset(('path', str(image_path)), image_path)],
                budget=ImagePayloadBudget(
                    max_images=1,
                    max_bytes=1,
                    max_pixels=100,
                ),
            )

        with pytest.raises(InputBudgetExceeded):
            await pack_content_batches(
                [[window]],
                project=project,
                budget=TokenBudget(
                    counter=MaterialTokenEstimator(_LengthCounter()),
                    token_limit=1,
                ),
                image_budget=ImagePayloadBudget(
                    max_images=0,
                    max_bytes=1,
                    max_pixels=1,
                ),
            )

    asyncio.run(run())


def test_metadata_text_budget_splits_whole_items_and_empty_context_skips_io():
    async def run() -> None:
        windows = [
            SourceBlockWindow(target=[_block(f'block-{index}')])
            for index in range(3)
        ]

        def project(items: list[SourceBlockWindow]) -> dict[str, object]:
            return {'evidence': [project_source_window(item) for item in items]}

        batches = await pack_content_batches(
            [windows],
            project=project,
            budget=TokenBudget(
                counter=MaterialTokenEstimator(_LengthCounter()),
                token_limit=200,
            ),
            image_budget=ImagePayloadBudget(
                max_images=0,
                max_bytes=1,
                max_pixels=1,
            ),
        )
        assert [len(batch) for batch in batches] == [1, 1, 1]

        assert (
            await load_content_images(
                [],
                budget=ImagePayloadBudget(
                    max_images=0,
                    max_bytes=1,
                    max_pixels=1,
                ),
            )
            == []
        )

    asyncio.run(run())


def test_ineligible_items_skip_projection_and_are_not_emitted(tmp_path):
    async def run() -> None:
        image_path = tmp_path / 'unneeded.png'
        _png(image_path, (255, 0, 0))
        window = SourceBlockWindow(
            target=[_block('one-card packet', str(image_path))]
        )
        projections: list[list[SourceBlockWindow]] = []

        def project(items: list[SourceBlockWindow]) -> dict[str, object]:
            projections.append(items)
            return {'evidence': [project_source_window(item) for item in items]}

        batches = await pack_content_batches(
            [[window]],
            project=project,
            budget=None,
            image_budget=ImagePayloadBudget(
                max_images=0,
                max_bytes=1,
                max_pixels=1,
            ),
            is_eligible=lambda items: len(items) >= 2,
        )

        assert batches == []
        assert projections == []

    asyncio.run(run())
