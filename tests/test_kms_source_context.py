import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import SourceBlockWindow
from kms.core.model.visual_asset import VisualAsset
from kms.core.source_context import count_source_tokens, project_source_window
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import select_window
from kms.service.content import load_content_images, pack_content_batches


def _png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new('RGB', (8, 8), color).save(path, format='PNG')


def _block(
    content: str | None,
    assets: list[VisualAsset] | None = None,
    block_type: BlockType = BlockType.PARAGRAPH,
) -> SourceBlock:
    return SourceBlock(
        block_type=block_type,
        content=content,
        assets=assets or [],
    )


def _pixel(payload: bytes) -> tuple[int, int, int]:
    with Image.open(BytesIO(payload)) as image:
        return image.getpixel((0, 0))


class _LengthCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def test_builder_deduplicates_exact_paths_and_preserves_asset_roles(tmp_path):
    red_path = tmp_path / 'red.png'
    blue_path = tmp_path / 'blue.png'
    green_path = tmp_path / 'green.png'
    _png(red_path, (255, 0, 0))
    _png(blue_path, (0, 0, 255))
    _png(green_path, (0, 255, 0))

    red_before = VisualAsset(uuid='red-before', path=str(red_path))
    red_after = VisualAsset(uuid='red-after', path=str(red_path))
    window = SourceBlockWindow(
        context_before=[_block('before', [red_before])],
        target=[
            _block(
                'figure',
                [
                    VisualAsset(path=str(blue_path)),
                    VisualAsset(path=str(green_path)),
                ],
                BlockType.IMAGE,
            )
        ],
        context_after=[_block('after', [red_after])],
    )

    builder = MultimodalContentBuilder()
    projected = project_source_window(window, builder)

    assert [str(image.payload) for image in builder.images] == [
        str(red_path),
        str(blue_path),
        str(green_path),
    ]
    assert projected.context_before[0].material.image_positions == [0]
    assert projected.target[0].material.image_positions == [1, 2]
    assert projected.context_after[0].material.image_positions == [0]
    assert len({asset.uuid for asset in (red_before, red_after)}) == 2
    assert str(red_path) not in projected.model_dump_json()
    assert 'red-before' not in projected.model_dump_json()
    assert _pixel(_read(str(builder.images[0].payload))) == (255, 0, 0)


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def test_source_window_costs_include_assets_but_not_crop_paths(tmp_path):
    image_path = tmp_path / 'figure.png'
    _png(image_path, (255, 0, 0))
    image = VisualAsset(path=str(image_path))
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
            assets=[image],
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
                block_type=BlockType.PARAGRAPH,
                content='x',
                crop_path=str(tmp_path / 'missing-mixed-crop.png'),
                assets=[image],
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
            SourceBlockWindow(
                target=[
                    _block('red paragraph', [VisualAsset(path=str(red_path))])
                ]
            ),
            SourceBlockWindow(
                target=[
                    _block('blue paragraph', [VisualAsset(path=str(blue_path))])
                ]
            ),
        ]

        def project(
            items: list[SourceBlockWindow], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            return {
                'evidence': [
                    project_source_window(item, builder) for item in items
                ]
            }

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
            builder = MultimodalContentBuilder()
            evidence = project(batch, builder)['evidence']
            assert evidence[0].target[0].material.image_positions == [0]
            images = await load_content_images(
                builder.images,
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
            SourceBlockWindow(
                target=[_block(path.stem, [VisualAsset(path=str(path))])]
            )
            for path in paths
        ]

        def project(
            items: list[SourceBlockWindow], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            return {
                'evidence': [
                    project_source_window(item, builder) for item in items
                ]
            }

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
        window = SourceBlockWindow(
            target=[_block('green', [VisualAsset(path=str(image_path))])]
        )

        def project(
            items: list[SourceBlockWindow], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            return {
                'evidence': [
                    project_source_window(item, builder) for item in items
                ]
            }

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
                [ContentImage(('path', str(image_path)), image_path)],
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

        def project(
            items: list[SourceBlockWindow], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            return {
                'evidence': [
                    project_source_window(item, builder) for item in items
                ]
            }

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
            target=[
                _block('one-card packet', [VisualAsset(path=str(image_path))])
            ]
        )
        projections: list[list[SourceBlockWindow]] = []

        def project(
            items: list[SourceBlockWindow], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            projections.append(items)
            return {
                'evidence': [
                    project_source_window(item, builder) for item in items
                ]
            }

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
