import asyncio
import uuid

import dspy

from kms2.config import ContextWindowSettings
from kms2.core.model import (
    BlockType,
    Source,
    SourceBlock,
    SourceBlockContext,
    SourcePage,
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitResult,
    VisualAsset,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source import splitter as splitter_module
from kms2.node.source.splitter import SplitterNode


class _Predictor:
    def __init__(self, **values: object) -> None:
        self.values = values
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(**self.values)

    async def acall(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(**self.values)


def _context(content: str, block_type: BlockType = BlockType.PARAGRAPH):
    return SourceBlockContext(block_type=block_type, content=content)


def _projected_parent(content: str) -> SourceBlockContext:
    return _context(
        content,
        BlockType.LIST,
    ).model_copy(update={'asset_paths': ['packed-figure.png']})


def test_signatures_expose_routed_split_contracts():
    assert set(splitter_module.ExerciseStripRouterSignature.input_fields) == {
        'context_before',
        'target_block',
        'context_after',
    }
    assert set(splitter_module.ExerciseStripRouterSignature.output_fields) == {
        'contains_multiple_exercises'
    }
    assert set(splitter_module.ExerciseSplitterSignature.input_fields) == {
        'context_before',
        'candidates',
        'context_after',
    }
    assert set(splitter_module.ExerciseSplitterSignature.output_fields) == {
        'splits'
    }


def test_router_forwards_context_and_returns_model_boolean():
    predictor = _Predictor(contains_multiple_exercises=True)
    router = splitter_module.ExerciseStripRouterModule(predictor)
    before = [_context('before')]
    target = _context('1. first 2. second')
    after = [_context('after')]

    assert (
        router(
            context_before=before,
            target_block=target,
            context_after=after,
        )
        is True
    )
    assert (
        asyncio.run(
            router.aforward(
                context_before=before,
                target_block=target,
                context_after=after,
            )
        )
        is True
    )
    assert predictor.calls == [
        {
            'context_before': before,
            'target_block': target,
            'context_after': after,
        },
        {
            'context_before': before,
            'target_block': target,
            'context_after': after,
        },
    ]


def test_splitter_forwards_candidates_and_returns_model_decisions():
    decision = SplitDecision(
        position=1,
        pieces=[SplitPiece(content='one'), SplitPiece(content='two')],
    )
    predictor = _Predictor(splits=[decision])
    splitter = splitter_module.ExerciseSplitterModule(predictor)
    before = [_context('before')]
    candidates = [
        SplitCandidate(position=1, source_block=_context('packed')),
    ]
    after = [_context('after')]

    assert splitter(
        context_before=before,
        candidates=candidates,
        context_after=after,
    ) == [decision]
    assert asyncio.run(
        splitter.aforward(
            context_before=before,
            candidates=candidates,
            context_after=after,
        )
    ) == [decision]
    assert predictor.calls == [
        {
            'context_before': before,
            'candidates': candidates,
            'context_after': after,
        },
        {
            'context_before': before,
            'candidates': candidates,
            'context_after': after,
        },
    ]


class _Router:
    def __init__(self, positive_contents: set[str]) -> None:
        self.positive_contents = positive_contents
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        target_block = kwargs['target_block']
        return target_block.content in self.positive_contents


class _Splitter:
    def __init__(self, decisions: list[SplitDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> list[SplitDecision]:
        self.calls.append(kwargs)
        return self.decisions


def _state(pages: list[SourcePage]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        image_enriched_pages=pages,
    )


def _node(
    router: _Router,
    splitter: _Splitter,
    *,
    target_budget: int = 0,
) -> SplitterNode:
    return SplitterNode(
        router,
        splitter,
        ContextWindowSettings(
            backward_budget=2,
            forward_budget=2,
            target_budget=target_budget,
        ),
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
                    block_type=BlockType.IMAGE, content='enriched image'
                ),
            ],
        ),
    ]
    node = _node(_Router(set()), _Splitter([]))

    sends = node.dispatch(_state(pages))

    assert isinstance(sends, list)
    assert [send.node for send in sends] == [
        'splitter_worker',
        'splitter_worker',
        'splitter_worker',
    ]
    requests = [send.arg['split_request'] for send in sends]
    assert [request.flat_position for request in requests] == [0, 1, 2]
    assert [request.window.target[0].content for request in requests] == [
        'first',
        'second',
        'enriched image',
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
    node = _node(_Router(set()), _Splitter([]), target_budget=3)

    sends = node.dispatch(_state(pages))
    requests = [send.arg['split_request'] for send in sends]

    assert [request.flat_position for request in requests] == [0, 2]
    assert [request.window.target for request in requests] == [
        [_context('one'), _context('two')],
        [_context('three')],
    ]
    assert [
        block.content for request in requests for block in request.window.target
    ] == ['one', 'two', 'three']


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
    sends = node.dispatch(_state([SourcePage(index=0, blocks=[first, second])]))

    result = asyncio.run(node.worker(sends[0].arg))

    assert [item.flat_position for item in result['split_results']] == [0, 1]
    assert [
        candidate.position for candidate in splitter.calls[0]['candidates']
    ] == [
        0,
        1,
    ]


def test_worker_returns_unsplit_result_without_calling_splitter():
    parent = SourceBlock(
        block_type=BlockType.LIST,
        content='ordinary',
    )
    node = _node(_Router(set()), _Splitter([]))
    sends = node.dispatch(_state([SourcePage(index=0, blocks=[parent])]))

    result = asyncio.run(node.worker(sends[0].arg))

    assert result == {
        'split_results': [SplitResult(flat_position=0, pieces=None)]
    }
    assert node._splitter.calls == []


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
    sends = node.dispatch(
        _state(
            [
                SourcePage(
                    index=0,
                    blocks=[
                        SourceBlock(
                            block_type=BlockType.PARAGRAPH, content='before'
                        ),
                        parent,
                    ],
                )
            ]
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
            'target_block': _projected_parent(parent.content),
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
