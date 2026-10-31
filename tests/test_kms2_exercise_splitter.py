from __future__ import annotations

import asyncio
import uuid

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitResult,
    SplitRouteDecision,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode


def _context(
    content: str, block_type: BlockType = BlockType.PARAGRAPH
) -> SourceBlockContext:
    return SourceBlockContext(block_type=block_type, content=content)


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

    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitRouteDecision]:
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
                    target.source_block.content in self.positive_contents
                ),
            )
            for target in reversed(targets)
        ]


class _Splitter:
    def __init__(self, decisions: list[SplitDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        context_after: list[SourceBlockContext],
    ) -> list[SplitDecision]:
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
    token_counters: tuple[_Counter, ...] | None = None,
) -> ExerciseSplitterNode:
    return ExerciseSplitterNode(
        router,
        exercise_splitter,
        ContextWindowSettings(
            backward_budget=2,
            forward_budget=2,
            target_budget=target_budget,
        ),
        batch_size=batch_size,
        token_counters=token_counters or (_Counter(),),
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
        token_counters=(_Counter([2, 2, 4]),),
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
    assert [request.window.context_before for request in requests] == [
        [],
        [_context('first')],
        [_context('second')],
    ]
    assert [request.window.context_after for request in requests] == [
        [_context('second')],
        [],
        [],
    ]
    for send in sends:
        asyncio.run(node.worker(send.arg))
    assert len(router.calls) == 3


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
    assert [request.window.target for request in requests] == [
        [_context('one'), _context('two')],
        [_context('three')],
    ]
    assert requests[0].window.context_after == [_context('three')]
    assert requests[1].window.context_before == [
        _context('one'),
        _context('two'),
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
        token_counters=(router_counter, splitter_counter),
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
        token_counters=(counter,),
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


def test_worker_splits_one_routed_candidate_at_local_position_zero():
    parent = SourceBlock(
        uuid='packed',
        block_type=BlockType.LIST,
        content='1. first exercise 2. second exercise',
        crop_path='packed.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[VisualAsset(path='packed-figure.png')],
    )
    router = _Router({parent.content})
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
    node = _node(router, splitter)
    sends = asyncio.run(
        node.dispatch(
            _state(
                [
                    SourcePage(
                        index=0,
                        blocks=[
                            SourceBlock(
                                block_type=BlockType.PARAGRAPH,
                                content='before',
                            ),
                            parent,
                        ],
                    )
                ]
            )
        )
    )

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
    assert router.calls == [
        {
            'context_before': [_context('before')],
            'targets': [
                PositionedSourceBlock(
                    position=0,
                    source_block=_projected_parent(parent.content),
                )
            ],
            'context_after': [],
        }
    ]
    assert splitter.calls == [
        {
            'context_before': [_context('before')],
            'candidates': [
                SplitCandidate(
                    position=0,
                    source_block=_projected_parent(parent.content),
                )
            ],
            'context_after': [],
        }
    ]


def test_collect_restores_order_and_preserves_unsplit_blocks():
    before = SourceBlock(
        uuid='before',
        block_type=BlockType.PARAGRAPH,
        content='before',
    )
    parent = SourceBlock(
        uuid='parent',
        block_type=BlockType.LIST,
        content='packed',
        crop_path='packed.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[VisualAsset(path='packed-figure.png')],
    )
    after = SourceBlock(
        uuid='after',
        block_type=BlockType.PARAGRAPH,
        content='after',
    )
    pages = [
        SourcePage(index=0, blocks=[before, parent]),
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
                ),
                SplitResult(
                    flat_position=1,
                    pieces=[
                        SplitPiece(content='first'),
                        SplitPiece(content='second'),
                    ],
                ),
            ]
        }
    )

    result = _node(_Router(set()), _Splitter([])).collect(state)
    split_pages = result['split_pages']
    child_blocks = split_pages[0].blocks[1:]

    assert split_pages[0].blocks[0] is before
    assert [block.content for block in child_blocks] == ['first', 'second']
    assert all(block.block_type is BlockType.LIST for block in child_blocks)
    assert all(block.uuid != parent.uuid for block in child_blocks)
    assert len({block.uuid for block in child_blocks}) == 2
    assert all(uuid.UUID(block.uuid).version == 4 for block in child_blocks)
    assert all(
        block.crop_path is None
        and block.crop_bbox is None
        and block.assets == []
        for block in child_blocks
    )
    assert [block.content for block in split_pages[1].blocks] == [
        'after one',
        'after two',
    ]
