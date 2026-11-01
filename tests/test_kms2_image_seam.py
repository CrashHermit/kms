import asyncio

from PIL import Image

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.image_seam import (
    ImageSeamDecision,
    ImageSeamInput,
    ImageSeamRequest,
    ImageSeamResult,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.image_seam import ImageSeamNode


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


class _Judge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def acall(
        self, *, items: list[ImageSeamInput], seam_images: list[object]
    ) -> list[ImageSeamDecision]:
        self.calls.append({'items': items, 'seam_images': seam_images})
        return [
            ImageSeamDecision(
                position=item.position,
                is_continuation=item.position == 0,
            )
            for item in reversed(items)
        ]


def _block(
    index: int,
    content: str | None = None,
    block_type: BlockType = BlockType.IMAGE,
    *,
    assets: list[VisualAsset] | None = None,
) -> SourceBlock:
    return SourceBlock(
        uuid=f'block-{index}',
        block_type=block_type,
        content=content,
        crop_path=f'crop-{index}.png',
        crop_bbox=(index, index + 1, index + 2, index + 3),
        assets=assets or [],
    )


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(uuid='source-1', key='document.pdf'),
        text_seam_pages=pages,
    )


def _node(judge: object, *, batch_size: int = 4) -> ImageSeamNode:
    return ImageSeamNode(
        judge,
        budget=TokenBudget(_Counter(), 10_000),
        batch_size=batch_size,
    )


def _request(index: int, top_path: str, bottom_path: str) -> ImageSeamRequest:
    return ImageSeamRequest(
        top_page=SourcePage(
            index=index * 2,
            blocks=[_block(index * 2, assets=[VisualAsset(path=top_path)])],
        ),
        bottom_page=SourcePage(
            index=index * 2 + 1,
            blocks=[
                _block(index * 2 + 1, assets=[VisualAsset(path=bottom_path)])
            ],
        ),
    )


def test_batch_worker_merges_only_continuation_and_preserves_metadata(tmp_path):
    paths = [tmp_path / f'image-{index}.png' for index in range(4)]
    for index, path in enumerate(paths):
        Image.new('RGB', (4, 4), color=(index * 30, 20, 10)).save(path)
    requests = [
        _request(0, str(paths[0]), str(paths[1])),
        _request(1, str(paths[2]), str(paths[3])),
    ]
    original = [request.model_copy(deep=True) for request in requests]
    judge = _Judge()
    results = asyncio.run(
        _node(judge).even_worker({'image_seam_requests': requests})
    )['image_seam_even_results']

    assert len(judge.calls) == 1
    assert [
        (item.position, item.top_image_position, item.bottom_image_position)
        for item in judge.calls[0]['items']
    ] == [(0, 0, 1), (1, 2, 3)]
    assert len(judge.calls[0]['seam_images']) == 4
    positive = results[0]
    assert [asset.path for asset in positive.top_page.blocks[0].assets] == [
        str(paths[0]),
        str(paths[1]),
    ]
    assert (
        positive.top_page.blocks[0].uuid == original[0].top_page.blocks[0].uuid
    )
    assert (
        positive.top_page.blocks[0].crop_path
        == original[0].top_page.blocks[0].crop_path
    )
    assert positive.bottom_page.blocks == []
    assert results[1].top_page == original[1].top_page
    assert results[1].bottom_page == original[1].bottom_page
    assert requests == original


def test_edge_selection_skips_apparatus_and_blocks_on_first_non_apparatus():
    judge = _Judge()
    node = _node(judge)
    valid = _block(0, assets=[VisualAsset(path='image.png')])
    pages = [
        SourcePage(
            index=0,
            blocks=[_block(1, 'bibliographic', BlockType.BIBLIOGRAPHIC), valid],
        ),
        SourcePage(
            index=1,
            blocks=[
                _block(2, 'note', BlockType.NOTE),
                _block(3, assets=[VisualAsset(path='bottom.png')]),
            ],
        ),
    ]
    sends = node.dispatch_even(_state(pages))
    assert isinstance(sends, list)
    assert len(sends) == 1
    assert sends[0].arg['image_seam_requests'][0].top_page.index == 0

    blocked_pages = [
        SourcePage(
            index=0, blocks=[valid, _block(4, 'text', BlockType.PARAGRAPH)]
        ),
        pages[1],
    ]
    assert (
        node.dispatch_even(_state(blocked_pages)) == 'image_seam_even_collect'
    )
    multiple_assets = SourcePage(
        index=1,
        blocks=[
            _block(
                5,
                assets=[
                    VisualAsset(path='bottom-one.png'),
                    VisualAsset(path='bottom-two.png'),
                ],
            )
        ],
    )
    assert (
        node.dispatch_even(
            _state([SourcePage(index=0, blocks=[valid]), multiple_assets])
        )
        == 'image_seam_even_collect'
    )


def test_physical_page_gaps_and_collectors_preserve_page_order():
    pages = [
        SourcePage(
            index=index,
            blocks=[_block(index, assets=[VisualAsset(path=f'{index}.png')])],
        )
        for index in (0, 2, 3, 5)
    ]
    node = _node(_Judge())
    sends = node.dispatch_even(_state(pages))
    assert isinstance(sends, list)
    pairs = [
        request for send in sends for request in send.arg['image_seam_requests']
    ]
    assert [
        (request.top_page.index, request.bottom_page.index) for request in pairs
    ] == [(2, 3)]
    contiguous_pages = [
        SourcePage(
            index=index,
            blocks=[_block(index, assets=[VisualAsset(path=f'{index}.png')])],
        )
        for index in range(4)
    ]
    singleton_sends = _node(_Judge(), batch_size=1).dispatch_even(
        _state(contiguous_pages)
    )
    assert isinstance(singleton_sends, list)
    assert len(singleton_sends) == 2
    assert [
        len(send.arg['image_seam_requests']) for send in singleton_sends
    ] == [1, 1]

    state = _state(pages)
    state.image_seam_even_results = [
        ImageSeamResult(
            top_page=pages[1].model_copy(update={'blocks': []}),
            bottom_page=pages[2].model_copy(update={'blocks': []}),
        )
    ]
    collected = node.even_collect(state)['image_seam_even_pages']
    assert [page.index for page in collected] == [0, 2, 3, 5]
    assert [len(page.blocks) for page in collected] == [1, 0, 0, 1]
