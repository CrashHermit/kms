from __future__ import annotations

import asyncio
import base64
import uuid
from io import BytesIO

from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms.core.model.page import SourcePage
from kms.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitResult,
    SplitRouteDecision,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode


def _context(
    content: str, block_type: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=block_type,
        material=MultimodalContentBuilder().build(text=content),
    )


def _projected_parent(content: str) -> SourceBlockContext:
    return _context(content, BlockType.LIST)


class _Counter:
    def __init__(self, counts: list[int] | None = None) -> None:
        self.counts = counts
        self.calls: list[list[str]] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls.append(texts)
        return self.counts if self.counts is not None else [1 for _ in texts]


class _Router:
    def __init__(self, positive_contents: set[str]) -> None:
        self.positive_contents = positive_contents
        self.calls: list[dict[str, object]] = []
        self.images: list[list] = []

    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
        content_images: list,
    ) -> list[SplitRouteDecision]:
        self.images.append(content_images)
        self.calls.append(
            {
                'context_before': context_before,
                'targets': targets,
                'context_after': context_after,
            }
        )
        return [
            SplitRouteDecision(
                position=target.position,
                contains_multiple_exercises=(
                    target.source_block.material.text in self.positive_contents
                ),
            )
            for target in reversed(targets)
        ]


class _Splitter:
    def __init__(self, decisions: list[SplitDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []
        self.reference_blocks: list[list[PositionedSourceBlock]] = []
        self.images: list[list] = []

    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        context_after: list[SourceBlockContext],
        reference_blocks: list[PositionedSourceBlock],
        content_images: list,
    ) -> list[SplitDecision]:
        self.reference_blocks.append(reference_blocks)
        self.images.append(content_images)
        self.calls.append(
            {
                'context_before': context_before,
                'candidates': candidates,
                'context_after': context_after,
            }
        )
        return self.decisions


def _state(pages: list[SourcePage]) -> SourcePedagogicalState:
    return SourcePedagogicalState(
        source_uuid='source-1',
        source_pages=pages,
    )


def _node(
    router: _Router,
    exercise_splitter: _Splitter,
    *,
    target_budget: int = 0,
    batch_size: int = 32,
    backward_budget: int = 2,
    token_counters: tuple[MaterialTokenEstimator, ...] | None = None,
) -> ExerciseSplitterNode:
    return ExerciseSplitterNode(
        router,
        exercise_splitter,
        ContextWindowSettings(
            backward_budget=backward_budget,
            forward_budget=2,
            target_budget=target_budget,
        ),
        batch_size=batch_size,
        token_counters=token_counters or (MaterialTokenEstimator(_Counter()),),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


def test_node_dispatches_one_context_window_per_flat_block():
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(block_type=BlockType.PARAGRAPH, content='first'),
                SourceBlock(block_type=BlockType.PARAGRAPH, content='second'),
            ],
        ),
        SourcePage(
            index=1,
            blocks=[
                SourceBlock(
                    block_type=BlockType.IMAGE, content='described image'
                ),
            ],
        ),
    ]
    router = _Router(set())
    node = _node(
        router,
        _Splitter([]),
        batch_size=1,
        token_counters=(MaterialTokenEstimator(_Counter([2, 2, 4])),),
    )

    sends = asyncio.run(node.dispatch(_state(pages)))

    assert isinstance(sends, list)
    assert [send.node for send in sends] == [
        'exercise_splitter_worker',
        'exercise_splitter_worker',
        'exercise_splitter_worker',
    ]
    requests = [send.arg['split_request'] for send in sends]
    assert [request.flat_position for request in requests] == [0, 1, 2]
    assert [request.window.target[0].content for request in requests] == [
        'first',
        'second',
        'described image',
    ]
    assert [
        [block.content for block in request.window.context_before]
        for request in requests
    ] == [[], ['first'], ['second']]
    assert [
        [block.content for block in request.window.context_after]
        for request in requests
    ] == [['second'], [], []]
    for send in sends:
        asyncio.run(node.worker(send.arg))
    assert len(router.calls) == 2


def test_node_dispatches_multiple_targets_when_target_budget_is_increased():
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(block_type=BlockType.PARAGRAPH, content='one'),
                SourceBlock(block_type=BlockType.PARAGRAPH, content='two'),
                SourceBlock(block_type=BlockType.PARAGRAPH, content='three'),
            ],
        )
    ]
    node = _node(
        _Router(set()),
        _Splitter([]),
        batch_size=2,
        target_budget=3,
    )

    sends = asyncio.run(node.dispatch(_state(pages)))
    requests = [send.arg['split_request'] for send in sends]

    assert [request.flat_position for request in requests] == [0, 2]
    assert [
        [block.content for block in request.window.target]
        for request in requests
    ] == [['one', 'two'], ['three']]
    assert [block.content for block in requests[0].window.context_after] == [
        'three'
    ]
    assert [block.content for block in requests[1].window.context_before] == [
        'one',
        'two',
    ]
    assert [
        block.content for request in requests for block in request.window.target
    ] == ['one', 'two', 'three']


def test_dispatch_uses_pointwise_maximum_of_two_profile_costs():
    blocks = [
        SourceBlock(block_type=BlockType.PARAGRAPH, content=content)
        for content in ('one', 'two', 'three')
    ]
    router_counter = _Counter([1, 4, 1])
    splitter_counter = _Counter([4, 1, 4])
    node = ExerciseSplitterNode(
        _Router(set()),
        _Splitter([]),
        ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
            target_budget=5,
        ),
        batch_size=32,
        token_counters=(
            MaterialTokenEstimator(router_counter),
            MaterialTokenEstimator(splitter_counter),
        ),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )

    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))
    )
    requests = [send.arg['split_request'] for send in sends]

    assert [request.flat_position for request in requests] == [0, 1, 2]
    assert [request.window.target[0].content for request in requests] == [
        'one',
        'two',
        'three',
    ]
    assert [request.window.context_before for request in requests] == [
        [],
        [],
        [],
    ]
    assert [request.window.context_after for request in requests] == [
        [],
        [],
        [],
    ]
    assert router_counter.calls == [['one', 'two', 'three']]
    assert splitter_counter.calls == [['one', 'two', 'three']]


def test_worker_returns_results_for_multiple_routed_targets():
    first = SourceBlock(block_type=BlockType.LIST, content='one')
    second = SourceBlock(block_type=BlockType.LIST, content='two')
    router = _Router({'one', 'two'})
    splitter = _Splitter(
        [
            SplitDecision(position=0, pieces=[SplitPiece(content='first')]),
            SplitDecision(position=1, pieces=[SplitPiece(content='second')]),
        ]
    )
    node = _node(router, splitter, target_budget=2)
    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=[first, second])]))
    )

    result = asyncio.run(node.worker(sends[0].arg))

    assert [item.flat_position for item in result['split_results']] == [0, 1]
    assert [
        candidate.position for candidate in splitter.calls[0]['candidates']
    ] == [0, 1]
    assert len(router.calls) == 1
    assert [target.position for target in router.calls[0]['targets']] == [0, 1]


def test_router_excludes_context_only_blocks_and_maps_local_positions():
    blocks = [
        SourceBlock(block_type=BlockType.LIST, content=content)
        for content in ('context only', 'packed exercise', 'numbered procedure')
    ]
    router = _Router({'packed exercise'})
    splitter = _Splitter(
        [
            SplitDecision(
                position=0,
                pieces=[
                    SplitPiece(content='Exercise one'),
                    SplitPiece(content='Exercise two'),
                ],
            )
        ]
    )
    node = _node(router, splitter, target_budget=1)
    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))
    )

    result = asyncio.run(node.worker(sends[1].arg))

    assert len(router.calls) == 1
    assert router.calls[0] == {
        'context_before': [_context('context only', BlockType.LIST)],
        'targets': [
            PositionedSourceBlock(
                position=0,
                source_block=_projected_parent('packed exercise'),
            )
        ],
        'context_after': [_context('numbered procedure', BlockType.LIST)],
    }
    assert splitter.calls[0]['candidates'] == [
        SplitCandidate(
            position=0,
            source_block=_projected_parent('packed exercise'),
        )
    ]
    assert result['split_results'] == [
        SplitResult(
            flat_position=1,
            pieces=[
                SplitPiece(content='Exercise one'),
                SplitPiece(content='Exercise two'),
            ],
        )
    ]


def test_empty_dispatch_collects_without_loading_counter():
    counter = _Counter()
    node = _node(
        _Router(set()),
        _Splitter([]),
        token_counters=(MaterialTokenEstimator(counter),),
    )

    assert asyncio.run(node.dispatch(_state([]))) == 'exercise_splitter_collect'
    assert counter.calls == []


def test_worker_returns_unsplit_result_without_calling_splitter():
    parent = SourceBlock(
        block_type=BlockType.LIST,
        content='ordinary',
    )
    node = _node(_Router(set()), _Splitter([]))
    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=[parent])]))
    )

    result = asyncio.run(node.worker(sends[0].arg))

    assert result == {
        'split_results': [SplitResult(flat_position=0, pieces=None)]
    }
    assert node._exercise_splitter.calls == []


def test_worker_splits_text_neighbor_and_preserves_visual_target(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (8, 8), (255, 0, 0)).save(image_path, format='PNG')
    visual = SourceBlock(
        uuid='figure-owner',
        block_type=BlockType.IMAGE,
        content='A diagram containing multiple exercises.',
        crop_path='figure-crop.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[VisualAsset(uuid='figure-asset', path=str(image_path))],
    )
    candidate = SourceBlock(
        uuid='text-list',
        block_type=BlockType.LIST,
        content='1. first exercise 2. second exercise',
    )
    router = _Router({visual.content, candidate.content})
    splitter = _Splitter(
        [
            SplitDecision(
                position=0,
                pieces=[
                    SplitPiece(content='1. first exercise'),
                    SplitPiece(content='2. second exercise'),
                ],
            )
        ]
    )
    node = _node(router, splitter, target_budget=1, backward_budget=3)
    state = _state([SourcePage(index=0, blocks=[visual, candidate])])
    sends = asyncio.run(node.dispatch(state))
    result = asyncio.run(node.worker(sends[1].arg))

    assert result['split_results'] == [
        SplitResult(
            flat_position=1,
            pieces=[
                SplitPiece(content='1. first exercise'),
                SplitPiece(content='2. second exercise'),
            ],
        )
    ]
    assert [target.position for target in router.calls[0]['targets']] == [0]
    assert router.calls[0]['context_before'][0].material.text == visual.content
    assert router.calls[0]['context_before'][0].material.image_positions == [0]
    assert len(router.images[0]) == 1
    router_payload = base64.b64decode(router.images[0][0].url.split(',', 1)[1])
    with Image.open(BytesIO(router_payload)) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    assert (
        splitter.calls[0]['context_before'][0].material.text == visual.content
    )
    assert splitter.calls[0]['context_before'][0].material.image_positions == [
        0
    ]
    assert splitter.calls[0]['candidates'] == [
        SplitCandidate(
            position=0,
            source_block=_projected_parent(candidate.content),
        )
    ]
    assert splitter.reference_blocks[0] == []
    assert len(splitter.images[0]) == 1
    splitter_payload = base64.b64decode(
        splitter.images[0][0].url.split(',', 1)[1]
    )
    with Image.open(BytesIO(splitter_payload)) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)

    state = state.model_copy(update={'split_results': result['split_results']})
    page = node.collect(state)['split_pages'][0]
    assert page.blocks[0] is visual
    assert page.blocks[0].assets == visual.assets
    assert page.blocks[0].crop_path == 'figure-crop.png'
    assert [block.content for block in page.blocks[1:]] == [
        '1. first exercise',
        '2. second exercise',
    ]


def test_collect_restores_order_and_preserves_unsplit_blocks():
    before = SourceBlock(
        uuid='before',
        block_type=BlockType.PARAGRAPH,
        content='before',
    )
    asset = VisualAsset(uuid='figure-asset', path='diagram-asset.png')
    visual = SourceBlock(
        uuid='visual-parent',
        block_type=BlockType.IMAGE,
        content='diagram',
        crop_path='diagram.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[asset],
    )
    after = SourceBlock(
        uuid='after',
        block_type=BlockType.LIST,
        content='after',
    )
    pages = [
        SourcePage(index=0, blocks=[before, visual]),
        SourcePage(index=1, blocks=[after]),
    ]
    state = _state(pages).model_copy(
        update={
            'split_results': [
                SplitResult(
                    flat_position=2,
                    pieces=[
                        SplitPiece(content='after one'),
                        SplitPiece(content='after two'),
                    ],
                )
            ]
        }
    )

    result = _node(_Router(set()), _Splitter([])).collect(state)
    split_pages = result['split_pages']
    child_blocks = split_pages[1].blocks

    assert split_pages[0].blocks[0] is before
    assert split_pages[0].blocks[1] is visual
    assert visual.uuid == 'visual-parent'
    assert visual.crop_path == 'diagram.png'
    assert visual.assets == [asset]
    assert [block.content for block in child_blocks] == [
        'after one',
        'after two',
    ]
    assert all(block.block_type is BlockType.LIST for block in child_blocks)
    assert all(block.uuid != after.uuid for block in child_blocks)
    assert len({block.uuid for block in child_blocks}) == 2
    assert all(uuid.UUID(block.uuid).version == 4 for block in child_blocks)
    assert {block.origin_uuid for block in child_blocks} == {after.uuid}
    repeated = _state(
        [
            SourcePage(
                index=1,
                blocks=[child_blocks[0]],
            )
        ]
    ).model_copy(
        update={
            'split_results': [
                SplitResult(
                    flat_position=0,
                    pieces=[SplitPiece(content='after one, part one')],
                )
            ]
        }
    )
    repeated_child = (
        _node(_Router(set()), _Splitter([]))
        .collect(repeated)['split_pages'][0]
        .blocks[0]
    )
    assert repeated_child.origin_uuid == child_blocks[0].uuid
