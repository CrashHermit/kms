import asyncio
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.images import fuse_image_blocks
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.processing.image_seam.models import (
    ImageSeamDecision,
    ImageSeamInput,
    ImageSeamResult,
)
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


class _Judge:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    async def acall(
        self, *, items: list[ImageSeamInput]
    ) -> list[ImageSeamDecision]:
        self.calls.append({'items': items})
        return [
            ImageSeamDecision(
                position=item.position,
                is_continuation=self.decisions[item.position],
            )
            for item in reversed(items)
        ]


def _page_image(path: Path, color: tuple[int, int, int]) -> None:
    with Image.new('RGB', (10, 10), 'white') as image:
        image.paste(color, (2, 8, 6, 10))
        image.save(path)


def _block(
    index: int,
    path: Path,
    bbox: tuple[int, int, int, int] = (2, 8, 6, 10),
) -> SourceBlock:
    return SourceBlock(
        uuid=f'block-{index}',
        block_type=BlockType.IMAGE,
        crop_path=str(path),
        crop_bbox=bbox,
        image_path=str(path),
        image_page_indexes=[index],
    )


def _state(
    pages: list[SourcePage], page_images: dict[int, str]
) -> SourceProcessingState:
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(
            uuid='00000000-0000-0000-0000-000000000001',
            key='document.pdf',
        ),
        text_seam_pages=pages,
        ocr_page_images=page_images,
    )


def _node(judge: object, *, batch_size: int = 4) -> ImageSeamNode:
    return ImageSeamNode(
        judge,
        budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
        image_budget=ImagePayloadBudget(
            max_images=8, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=batch_size,
    )


def _pages(
    tmp_path: Path, indexes: tuple[int, ...] = (4, 5, 6)
) -> tuple[list[SourcePage], dict[int, str]]:
    pages = []
    page_images = {}
    for index in indexes:
        page_path = tmp_path / f'page-{index}.png'
        crop_path = tmp_path / f'crop-{index}.png'
        color = (index * 20, 30, 40)
        _page_image(page_path, color)
        _page_image(crop_path, color)
        page_images[index] = str(page_path)
        pages.append(SourcePage(index=index, blocks=[_block(index, crop_path)]))
    return pages, page_images


def _dispatch_and_collect(
    node: ImageSeamNode, state: SourceProcessingState
) -> list[SourcePage]:
    sends = asyncio.run(node.dispatch(state))
    assert isinstance(sends, list)
    results = []
    for send in sends:
        results.extend(asyncio.run(node.worker(send.arg))['image_seam_results'])
    state.image_seam_results = results
    return asyncio.run(node.collect(state))['image_seam_pages']


def test_three_original_adjacent_seams_fuse_one_positioned_image(tmp_path):
    pages, page_images = _pages(tmp_path)
    first_crop_path = pages[0].blocks[0].crop_path
    assert first_crop_path is not None
    first_crop = Path(first_crop_path).read_bytes()
    node = _node(_Judge([True, True]))

    collected = _dispatch_and_collect(node, _state(pages, page_images))

    assert [page.index for page in collected] == [4, 5, 6]
    assert len(collected[0].blocks) == 1
    assert [page.blocks for page in collected[1:]] == [[], []]
    fused = collected[0].blocks[0]
    assert fused.uuid == pages[0].blocks[0].uuid
    assert fused.crop_path == pages[0].blocks[0].crop_path
    assert fused.crop_bbox == pages[0].blocks[0].crop_bbox
    assert fused.image_page_indexes == [4, 5, 6]
    assert fused.image_sha256 is None
    fused_image_path = fused.image_path
    assert fused_image_path is not None
    with Image.open(fused_image_path) as image:
        assert image.size == (4, 22)
        assert image.getpixel((0, 0)) == (80, 30, 40)
        assert image.getpixel((0, 10)) == (100, 30, 40)
        assert image.getpixel((0, 20)) == (120, 30, 40)
    assert Path(first_crop_path).read_bytes() == first_crop


def test_negative_edge_splits_chains(tmp_path):
    pages, page_images = _pages(tmp_path)
    node = _node(_Judge([True, False]))
    collected = _dispatch_and_collect(node, _state(pages, page_images))

    assert len(collected[0].blocks) == 1
    assert collected[0].blocks[0].image_page_indexes == [4, 5]
    assert collected[1].blocks == []
    assert len(collected[2].blocks) == 1
    assert collected[2].blocks[0].image_page_indexes == [6]


def test_separate_figures_on_intervening_page_form_distinct_chains(tmp_path):
    page_images = {}
    for index in range(3):
        path = tmp_path / f'page-{index}.png'
        _page_image(path, (index * 20, 30, 40))
        page_images[index] = str(path)

    def block(
        name: str,
        page: int,
        bbox: tuple[int, int, int, int],
    ) -> SourceBlock:
        crop = tmp_path / f'{name}.png'
        _page_image(crop, (page * 20, 30, 40))
        return SourceBlock(
            uuid=name,
            block_type=BlockType.IMAGE,
            crop_path=str(crop),
            crop_bbox=bbox,
            image_path=str(crop),
            image_page_indexes=[page],
        )

    pages = [
        SourcePage(index=0, blocks=[block('first', 0, (2, 8, 6, 10))]),
        SourcePage(
            index=1,
            blocks=[
                block('middle-top', 1, (2, 0, 6, 2)),
                block('middle-bottom', 1, (2, 8, 6, 10)),
            ],
        ),
        SourcePage(index=2, blocks=[block('last', 2, (2, 8, 6, 10))]),
    ]

    collected = _dispatch_and_collect(
        _node(_Judge([True, True])), _state(pages, page_images)
    )

    assert [block.uuid for page in collected for block in page.blocks] == [
        'first',
        'middle-bottom',
    ]
    assert [
        block.image_page_indexes for page in collected for block in page.blocks
    ] == [[0, 1], [1, 2]]
    assert collected[2].blocks == []


def test_dispatch_judges_only_physical_adjacent_image_edges(tmp_path):
    pages, page_images = _pages(tmp_path, indexes=(0, 2, 3, 5))
    sends = asyncio.run(
        _node(_Judge([True])).dispatch(_state(pages, page_images))
    )
    assert isinstance(sends, list)
    requests = [
        request for send in sends for request in send.arg['image_seam_requests']
    ]
    assert [
        (item.top_page.index, item.bottom_page.index) for item in requests
    ] == [(2, 3)]
    assert [item.position for item in requests] == [0]


def test_worker_reuses_catalog_positions_and_maps_reordered_decisions(tmp_path):
    pages, page_images = _pages(tmp_path, indexes=(0, 1, 2))
    shared_path = pages[1].blocks[0].image_path
    shared = pages[0].blocks[0].model_copy(update={'image_path': shared_path})
    pages[0] = pages[0].model_copy(update={'blocks': [shared]})
    judge = _Judge([True, False])
    node = _node(judge, batch_size=1)
    state = _state(pages, page_images)

    sends = asyncio.run(node.dispatch(state))
    assert isinstance(sends, list)
    assert len(sends) == 2
    first_result = asyncio.run(node.worker(sends[0].arg))['image_seam_results'][
        0
    ]
    assert first_result == ImageSeamResult(position=0, is_continuation=True)
    first_call = judge.calls[0]
    items = first_call['items']
    assert items[0].top.images == items[0].bottom.images


def test_fusion_over_budget_does_not_publish_an_artifact(tmp_path):
    pages, page_images = _pages(tmp_path, indexes=(0, 1))
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            fuse_image_blocks(
                [(page.index, page.blocks[0]) for page in pages],
                page_images=page_images,
                budget=ImagePayloadBudget(
                    max_images=4,
                    max_bytes=1_000_000,
                    max_pixels=1,
                ),
            )
        )
    assert not list(tmp_path.glob('Fused_*.png'))


def test_run_blocking_work_waits_for_blocking_work_before_cancellation_returns():
    finished = threading.Event()

    def blocking_work() -> None:
        time.sleep(0.02)
        finished.set()

    async def cancel_during_work() -> None:
        task = asyncio.create_task(run_blocking_work(blocking_work))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            assert finished.is_set()
        else:
            raise AssertionError('Cancellation was not propagated')

    asyncio.run(cancel_during_work())
