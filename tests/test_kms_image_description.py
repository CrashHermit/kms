import asyncio

import dspy
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionOutput,
    ImageDescriptionResult,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.node.source_processing.image_description import ImageDescriptionNode


class _Counter:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls.append(texts)
        return [1 for _ in texts]


class _Describer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def acall(
        self,
        *,
        items: list[ImageDescriptionInput],
        content_images: list[dspy.Image],
    ) -> list[ImageDescriptionOutput]:
        self.calls.append({'items': items, 'content_images': content_images})
        return [
            ImageDescriptionOutput(
                position=item.position,
                description=f'description {item.position}',
            )
            for item in reversed(items)
        ]


def _block(
    *,
    block_type: BlockType,
    content: str | None = None,
    assets: list[str] | None = None,
) -> SourceBlock:
    return SourceBlock(
        block_type=block_type,
        content=content,
        assets=[VisualAsset(path=path) for path in assets or []],
    )


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(uuid='source-1', key='document.pdf'),
        image_filtered_pages=pages,
    )


def _node(
    describer: object,
    *,
    counter: _Counter | None = None,
    batch_size: int = 4,
) -> ImageDescriptionNode:
    token_counter = counter or _Counter()
    return ImageDescriptionNode(
        describer,
        ContextWindowSettings(backward_budget=200, forward_budget=200),
        budget=TokenBudget(MaterialTokenEstimator(token_counter), 10_000),
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=batch_size,
        token_counters=(MaterialTokenEstimator(token_counter),),
    )


def test_worker_batches_logical_images_and_preserves_ordered_asset_groups(
    tmp_path,
):
    paths = [tmp_path / f'image-{index}.png' for index in range(3)]
    for index, path in enumerate(paths):
        Image.new('RGB', (3, 3), color=(index * 40, 10, 20)).save(path)
    first = _block(
        block_type=BlockType.IMAGE,
        assets=[str(paths[0]), str(paths[1])],
    )
    second = _block(block_type=BlockType.IMAGE, assets=[str(paths[2])])
    before = _block(block_type=BlockType.PARAGRAPH, content='before target one')
    after = _block(block_type=BlockType.PARAGRAPH, content='after target two')
    pages = [
        SourcePage(index=0, blocks=[before, first]),
        SourcePage(index=1, blocks=[second, after]),
    ]
    describer = _Describer()
    node = _node(describer)

    sends = asyncio.run(node.dispatch(_state(pages)))
    assert isinstance(sends, list)
    assert len(sends) == 1
    requests = sends[0].arg['image_description_requests']
    assert [request.flat_position for request in requests] == [1, 2]
    assert [item.content for item in requests[0].window.context_before] == [
        'before target one'
    ]
    assert [item.content for item in requests[1].window.context_after] == [
        'after target two'
    ]
    results = asyncio.run(node.worker({'image_description_requests': requests}))
    singleton_sends = asyncio.run(
        _node(object(), batch_size=1).dispatch(_state(pages))
    )
    assert isinstance(singleton_sends, list)
    assert len(singleton_sends) == 2
    assert [
        len(send.arg['image_description_requests']) for send in singleton_sends
    ] == [1, 1]

    call = describer.calls[0]
    assert [item.target.image_positions for item in call['items']] == [
        [0, 1],
        [2],
    ]
    assert call['items'][0].context_after[0].material.image_positions == [2]
    assert call['items'][1].context_before[-1].material.image_positions == [
        0,
        1,
    ]
    assert [item.position for item in call['items']] == [0, 1]
    assert len(call['content_images']) == 3
    assert all(
        isinstance(image, dspy.Image) for image in call['content_images']
    )
    assert [
        result.description for result in results['image_description_results']
    ] == [
        'description 0',
        'description 1',
    ]


def test_dispatch_targets_only_images_with_available_assets(tmp_path):
    target_path = tmp_path / 'target.png'
    Image.new('RGB', (3, 3), color=(10, 20, 30)).save(target_path)
    pages = [
        SourcePage(
            index=0,
            blocks=[
                _block(block_type=BlockType.PARAGRAPH, content='before'),
                _block(
                    block_type=BlockType.IMAGE,
                    assets=[str(target_path)],
                ),
                _block(block_type=BlockType.IMAGE, content='nearby image'),
                _block(block_type=BlockType.IMAGE),
                _block(block_type=BlockType.PARAGRAPH, content='after'),
            ],
        )
    ]
    node = _node(object())
    state = _state(pages)
    sends = asyncio.run(node.dispatch(state))
    assert isinstance(sends, list)
    requests = [
        request
        for send in sends
        for request in send.arg['image_description_requests']
    ]
    assert [request.flat_position for request in requests] == [1]
    first = requests[0]
    assert [item.content for item in first.window.context_before] == ['before']
    assert [item.content for item in first.window.context_after] == [
        'nearby image',
        None,
        'after',
    ]
    assert requests[0].source_block.assets
    state.image_description_results = [
        ImageDescriptionResult(
            flat_position=1, description='target description'
        )
    ]
    preserved = node.collect(state)['image_described_pages'][0].blocks
    assert preserved[2] is pages[0].blocks[2]
    assert preserved[3] is pages[0].blocks[3]


def test_no_images_route_directly_to_collect_without_token_counting():
    counter = _Counter()
    node = _node(object(), counter=counter)
    state = _state(
        [
            SourcePage(
                index=0,
                blocks=[_block(block_type=BlockType.PARAGRAPH, content='x')],
            )
        ]
    )

    assert asyncio.run(node.dispatch(state)) == 'image_description_collect'
    assert counter.calls == []


def test_collect_applies_results_without_losing_asset_or_page_metadata():
    first = _block(block_type=BlockType.PARAGRAPH, content='first')
    image = _block(block_type=BlockType.IMAGE, assets=['target.png'])
    second = _block(block_type=BlockType.PARAGRAPH, content='second')
    pages = [
        SourcePage(index=0, blocks=[first, image]),
        SourcePage(index=1, blocks=[second]),
    ]
    state = _state(pages).model_copy(
        update={
            'image_description_results': [
                ImageDescriptionResult(
                    flat_position=1, description='description'
                )
            ]
        }
    )

    described_pages = _node(object()).collect(state)['image_described_pages']
    assert [page.index for page in described_pages] == [0, 1]
    assert described_pages[0].blocks[0] is first
    assert described_pages[0].blocks[1].content == 'description'
    assert described_pages[0].blocks[1].uuid == image.uuid
    assert described_pages[0].blocks[1].assets == image.assets
    assert described_pages[1].blocks[0] is second
