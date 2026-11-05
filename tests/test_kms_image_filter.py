import asyncio

import dspy
import pytest
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.image_filter import (
    ImageFilterDecision,
    ImageFilterInput,
    ImageFilterResult,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.node.source_processing.image_filter import ImageFilterNode


class _Counter:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def count_texts(self, texts: list[str]) -> list[int]:
        return [self.value for _ in texts]


class _Filter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ImageFilterInput], list[dspy.Image]]] = []

    async def acall(
        self,
        *,
        items: list[ImageFilterInput],
        content_images: list[dspy.Image],
    ) -> list[ImageFilterDecision]:
        self.calls.append((items, content_images))
        return [
            ImageFilterDecision(
                position=item.position,
                keep=any(
                    'study this advertisement' in (context.material.text or '')
                    for context in (*item.context_before, *item.context_after)
                ),
            )
            for item in reversed(items)
        ]


def _block(
    block_type: BlockType,
    content: str | None = None,
    paths: list[str] | None = None,
) -> SourceBlock:
    return SourceBlock(
        block_type=block_type,
        content=content,
        assets=[VisualAsset(path=path) for path in paths or []],
    )


def _state(pages: list[SourcePage], results=None) -> SourceProcessingState:
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(uuid='source-1', key='document.pdf'),
        image_seam_pages=pages,
        image_filter_results=results or [],
    )


def _node(
    module: object,
    *,
    counter: _Counter | None = None,
    token_limit: int = 100_000,
    image_budget: ImagePayloadBudget | None = None,
) -> ImageFilterNode:
    wrapped = MaterialTokenEstimator(counter or _Counter())
    return ImageFilterNode(
        module,
        ContextWindowSettings(backward_budget=100, forward_budget=100),
        budget=TokenBudget(wrapped, token_limit),
        image_budget=image_budget
        or ImagePayloadBudget(
            max_images=20, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=4,
        token_counters=(wrapped,),
    )


def _png(path, size=(8, 8)) -> str:
    Image.new('RGB', size, (10, 20, 30)).save(path)
    return str(path)


def test_worker_projects_target_context_and_maps_local_results(
    tmp_path,
):
    first_path = _png(tmp_path / 'first.png')
    second_path = _png(tmp_path / 'second.png')
    nearby_path = _png(tmp_path / 'nearby.png')
    text = _block(
        BlockType.PARAGRAPH, 'please study this advertisement carefully.'
    )
    target = _block(BlockType.IMAGE, paths=[first_path, second_path])
    nearby = _block(BlockType.IMAGE, paths=[nearby_path])
    pages = [
        SourcePage(
            index=0, markdown='raw capture', blocks=[text, target, nearby]
        )
    ]
    state = _state(pages)
    module = _Filter()
    node = _node(module)

    sends = asyncio.run(node.dispatch(state))
    assert isinstance(sends, list) and len(sends) == 1
    requests = sends[0].arg['image_filter_requests']
    assert [request.flat_position for request in requests] == [1, 2]
    results = asyncio.run(node.worker({'image_filter_requests': requests}))
    assert [
        result.flat_position for result in results['image_filter_results']
    ] == [1, 2]
    items, images = module.calls[0]
    assert [item.position for item in items] == [0, 1]
    assert items[0].target.image_positions == [0, 1]
    assert items[1].target.image_positions == [2]
    assert items[0].context_before[0].material.text == text.content
    assert items[1].context_before[0].material.text == text.content
    assert len(images) == 3 and all(
        isinstance(image, dspy.Image) for image in images
    )
    assert [result.keep for result in results['image_filter_results']] == [
        True,
        True,
    ]


def test_ad_is_kept_when_studied_and_rejected_without_that_context(tmp_path):
    path = _png(tmp_path / 'ad.png')
    target = _block(BlockType.IMAGE, paths=[path])
    module = _Filter()
    node = _node(module)
    decisions = []
    for text in (
        'please study this advertisement as persuasive design.',
        'A paragraph about limits and continuity.',
    ):
        state = _state(
            [
                SourcePage(
                    index=0,
                    blocks=[
                        _block(BlockType.PARAGRAPH, text),
                        target,
                    ],
                )
            ]
        )
        sends = asyncio.run(node.dispatch(state))
        requests = sends[0].arg['image_filter_requests']
        result = asyncio.run(node.worker({'image_filter_requests': requests}))
        decisions.append(result['image_filter_results'][0].keep)
    assert decisions == [True, False]


def test_collect_preserves_pages_and_removes_only_rejected_images():
    first = _block(BlockType.PARAGRAPH, 'first')
    rejected = _block(BlockType.IMAGE, paths=['capture-ad.png'])
    retained = _block(BlockType.IMAGE, paths=['figure.png', 'figure-part.png'])
    final = _block(BlockType.PARAGRAPH, 'last')
    last_rejected = _block(BlockType.IMAGE, paths=['another-ad.png'])
    pages = [
        SourcePage(
            index=4, markdown='original markdown', blocks=[first, rejected]
        ),
        SourcePage(index=9, markdown='empty capture', blocks=[retained, final]),
        SourcePage(
            index=12,
            markdown='all image content rejected',
            blocks=[last_rejected],
        ),
    ]
    state = _state(
        pages,
        [
            ImageFilterResult(flat_position=1, keep=False),
            ImageFilterResult(flat_position=2, keep=True),
            ImageFilterResult(flat_position=4, keep=False),
        ],
    )

    filtered = _node(object()).collect(state)['image_filtered_pages']
    assert [(page.index, page.markdown) for page in filtered] == [
        (4, 'original markdown'),
        (9, 'empty capture'),
        (12, 'all image content rejected'),
    ]
    assert filtered[0].blocks == [first]
    assert filtered[1].blocks == [retained, final]
    assert filtered[1].blocks[0].uuid == retained.uuid
    assert filtered[1].blocks[0].assets == retained.assets
    assert filtered[1].blocks[0].assets[0].path == 'figure.png'
    assert filtered[2].blocks == []


def test_assetless_and_no_image_inputs_skip_inference():
    counter = _Counter()
    module = _Filter()
    node = _node(module, counter=counter)
    blocks = [
        [_block(BlockType.IMAGE)],
        [_block(BlockType.PARAGRAPH, 'text-only')],
    ]
    for page_blocks in blocks:
        state = _state([SourcePage(index=0, blocks=page_blocks)])
        assert asyncio.run(node.dispatch(state)) == 'image_filter_collect'
        assert (
            node.collect(state)['image_filtered_pages'][0].blocks == page_blocks
        )
    assert counter.value == 0 and module.calls == []


def test_combined_token_and_media_overflow_fail_before_model_call(tmp_path):
    path = _png(tmp_path / 'target.png', (64, 64))
    target = _block(BlockType.IMAGE, paths=[path])
    state = _state([SourcePage(index=0, blocks=[target])])
    module = _Filter()
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(_node(module, token_limit=1).dispatch(state))
    assert module.calls == []

    with pytest.raises(InputBudgetExceeded, match='payload budget'):
        asyncio.run(
            _node(
                module,
                image_budget=ImagePayloadBudget(
                    max_images=0, max_bytes=25_000_000, max_pixels=40_000_000
                ),
            ).dispatch(state)
        )
    assert module.calls == []


def test_missing_native_image_propagates_before_model_call(tmp_path):
    target = _block(BlockType.IMAGE, paths=[str(tmp_path / 'missing.png')])
    state = _state([SourcePage(index=0, blocks=[target])])
    module = _Filter()
    with pytest.raises(FileNotFoundError):
        asyncio.run(_node(module).dispatch(state))
    assert module.calls == []
