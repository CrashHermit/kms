import asyncio

import pytest

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.text_seam import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRequest,
    TextSeamResult,
    TextSeamRewriteOutput,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.text_seam import TextSeamNode


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls += 1
        return [1 for _ in texts]


def _block(
    index: int,
    content: str,
    block_type: BlockType = BlockType.PARAGRAPH,
    *,
    assets: list[VisualAsset] | None = None,
) -> SourceBlock:
    return SourceBlock(
        uuid=f'block-{index}-{content}',
        block_type=block_type,
        content=content,
        crop_path=f'crop-{index}.png',
        crop_bbox=(index, index + 1, index + 2, index + 3),
        assets=assets or [],
    )


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    return SourceProcessingState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        formatted_pages=pages,
    )


class _Judge:
    def __init__(self) -> None:
        self.calls: list[list[TextSeamInput]] = []

    async def acall(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamDecision]:
        self.calls.append(items)
        return [
            TextSeamDecision(
                position=item.position,
                is_split=item.tail.content in {'tail-0', 'tail-2'},
            )
            for item in reversed(items)
        ]


class _Rewriter:
    def __init__(self) -> None:
        self.calls: list[list[TextSeamInput]] = []

    async def acall(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamRewriteOutput]:
        self.calls.append(items)
        return [
            TextSeamRewriteOutput(
                position=item.position, merged=f'merged-{item.position}'
            )
            for item in reversed(items)
        ]


def _node(judge: object, rewriter: object) -> TextSeamNode:
    return TextSeamNode(
        judge,
        rewriter,
        judge_budget=TokenBudget(_Counter(), 10_000),
        rewrite_budget=TokenBudget(_Counter(), 10_000),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )


def _pair(index: int) -> TextSeamRequest:
    top_context = _block(index * 2, f'context-top-{index}')
    tail = _block(index * 2 + 1, f'tail-{index}')
    head = _block(index * 2 + 2, f'head-{index}')
    bottom_context = _block(index * 2 + 3, f'context-bottom-{index}')
    return TextSeamRequest(
        top_page=SourcePage(index=index * 2, blocks=[top_context, tail]),
        bottom_page=SourcePage(
            index=index * 2 + 1, blocks=[head, bottom_context]
        ),
    )


def test_positive_only_sparse_rewrites_preserve_tail_provenance_and_negative_pages():
    requests = [_pair(index) for index in range(3)]
    original = [request.model_copy(deep=True) for request in requests]
    judge = _Judge()
    rewriter = _Rewriter()
    results = asyncio.run(
        _node(judge, rewriter).even_worker({'text_seam_requests': requests})
    )['text_seam_even_results']

    assert len(judge.calls) == 1
    assert [item.position for item in judge.calls[0]] == [0, 1, 2]
    assert len(rewriter.calls) == 1
    assert [item.position for item in rewriter.calls[0]] == [0, 2]
    assert [result.top_page.blocks[-1].content for result in results] == [
        'merged-0',
        'tail-1',
        'merged-2',
    ]
    assert [len(result.bottom_page.blocks) for result in results] == [1, 2, 1]
    for position in (0, 2):
        assert (
            results[position].top_page.blocks[-1].uuid
            == original[position].top_page.blocks[-1].uuid
        )
        assert (
            results[position].top_page.blocks[-1].crop_path
            == original[position].top_page.blocks[-1].crop_path
        )
        assert (
            results[position].top_page.blocks[-1].assets
            == original[position].top_page.blocks[-1].assets
        )
    assert requests == original


def test_text_edge_selection_skips_visual_apparatus_and_assets():
    top = SourcePage(
        index=0,
        blocks=[
            _block(0, 'top context'),
            _block(1, 'bibliographic', BlockType.BIBLIOGRAPHIC),
            _block(2, 'image', BlockType.IMAGE),
            _block(3, 'asset', assets=[VisualAsset(path='asset.png')]),
            _block(4, 'tail'),
        ],
    )
    bottom = SourcePage(
        index=1,
        blocks=[
            _block(5, 'head'),
            _block(6, 'image', BlockType.IMAGE),
            _block(7, 'note', BlockType.NOTE),
            _block(8, 'asset after', assets=[VisualAsset(path='after.png')]),
            _block(9, 'bottom context'),
        ],
    )
    judge = _Judge()
    asyncio.run(
        _node(judge, _Rewriter()).even_worker(
            {
                'text_seam_requests': [
                    TextSeamRequest(top_page=top, bottom_page=bottom)
                ]
            }
        )
    )
    assert judge.calls[0][0].tail.content == 'tail'
    assert judge.calls[0][0].head.content == 'head'
    assert judge.calls[0][0].top_context.content == 'top context'
    assert judge.calls[0][0].bottom_context.content == 'bottom context'

    empty_top = SourcePage(index=0, blocks=[_block(11, 'tail'), _block(10, '')])
    empty_bottom = SourcePage(
        index=1, blocks=[_block(12, ''), _block(13, 'head')]
    )
    counter = _Counter()
    node = TextSeamNode(
        judge,
        _Rewriter(),
        judge_budget=TokenBudget(counter, 10_000),
        rewrite_budget=TokenBudget(counter, 10_000),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )
    assert (
        node.dispatch_even(_state([empty_top, empty_bottom]))
        == 'text_seam_even_collect'
    )


def test_physical_page_gaps_block_candidates_and_collectors_preserve_order():
    pages = [
        SourcePage(index=index, blocks=[_block(index, f'page {index}')])
        for index in (0, 2, 3, 5)
    ]
    node = _node(_Judge(), _Rewriter())
    sends = node.dispatch_even(_state(pages))
    assert isinstance(sends, list)
    pairs = [
        request for send in sends for request in send.arg['text_seam_requests']
    ]
    assert [
        (pair.top_page.index, pair.bottom_page.index) for pair in pairs
    ] == [(2, 3)]

    all_pages = [
        SourcePage(index=index, blocks=[_block(index, f'page {index}')])
        for index in range(5)
    ]
    state = _state(all_pages)
    state.text_seam_even_results = [
        TextSeamResult(
            top_page=all_pages[2].model_copy(
                update={'blocks': [_block(2, 'merged 2')]}
            ),
            bottom_page=all_pages[3].model_copy(
                update={'blocks': [_block(30, 'remaining 3')]}
            ),
        ),
        TextSeamResult(
            top_page=all_pages[0].model_copy(
                update={'blocks': [_block(0, 'merged 0')]}
            ),
            bottom_page=all_pages[1].model_copy(update={'blocks': []}),
        ),
    ]
    collected = node.even_collect(state)['text_seam_even_pages']
    assert [page.index for page in collected] == [0, 1, 2, 3, 4]
    assert [
        page.blocks[0].content if page.blocks else None for page in collected
    ] == ['merged 0', None, 'merged 2', 'remaining 3', 'page 4']

    odd_state = state.model_copy(update={'text_seam_even_pages': collected})
    odd_sends = node.dispatch_odd(odd_state)
    assert isinstance(odd_sends, list)
    odd_pairs = [
        request
        for send in odd_sends
        for request in send.arg['text_seam_requests']
    ]
    assert [
        (pair.top_page.index, pair.bottom_page.index) for pair in odd_pairs
    ] == [(3, 4)]
    assert odd_pairs[0].top_page.blocks[0].content == 'remaining 3'


def test_no_positive_judgments_never_calls_rewriter():
    class _NegativeJudge:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamDecision]:
            return [
                TextSeamDecision(position=item.position, is_split=False)
                for item in items
            ]

    rewriter = _Rewriter()
    results = asyncio.run(
        _node(_NegativeJudge(), rewriter).even_worker(
            {'text_seam_requests': [_pair(0)]}
        )
    )
    assert (
        results['text_seam_even_results'][0].top_page.blocks[-1].content
        == 'tail-0'
    )
    assert rewriter.calls == []


def test_positive_merge_keeps_empty_page_after_removing_only_head():
    class _PositiveJudge:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamDecision]:
            return [
                TextSeamDecision(position=item.position, is_split=True)
                for item in items
            ]

    class _FixedRewriter:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamRewriteOutput]:
            return [
                TextSeamRewriteOutput(position=item.position, merged='joined')
                for item in items
            ]

    request = TextSeamRequest(
        top_page=SourcePage(index=0, blocks=[_block(20, 'tail')]),
        bottom_page=SourcePage(index=1, blocks=[_block(21, 'head')]),
    )
    result = asyncio.run(
        _node(_PositiveJudge(), _FixedRewriter()).even_worker(
            {'text_seam_requests': [request]}
        )
    )['text_seam_even_results'][0]
    assert result.top_page.blocks[0].content == 'joined'
    assert result.bottom_page.blocks == []


@pytest.mark.parametrize(
    ('token_limit', 'packet_sizes'),
    [(6, [1, 1]), (7, [2])],
)
def test_token_budget_packs_whole_pairs_at_the_inclusive_limit(
    token_limit, packet_sizes
):
    class _PayloadCounter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [
                1 if text in ('{"items":[]}', '[,]') else 2 for text in texts
            ]

    pages = [
        SourcePage(index=index, blocks=[_block(index, f'page {index}')])
        for index in range(4)
    ]
    counter = _PayloadCounter()
    node = TextSeamNode(
        _Judge(),
        _Rewriter(),
        judge_budget=TokenBudget(counter, token_limit),
        rewrite_budget=TokenBudget(counter, 10_000),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )
    sends = node.dispatch_even(_state(pages))
    assert isinstance(sends, list)
    assert [
        len(send.arg['text_seam_requests']) for send in sends
    ] == packet_sizes


def test_oversized_atomic_seam_request_raises_without_fallback_calls():
    class _OversizeCounter:
        def count_texts(self, texts: list[str]) -> list[int]:
            return [
                1 if text in ('{"items":[]}', '[,]') else 100 for text in texts
            ]

    judge = _Judge()
    rewriter = _Rewriter()
    node = TextSeamNode(
        judge,
        rewriter,
        judge_budget=TokenBudget(_OversizeCounter(), 10),
        rewrite_budget=TokenBudget(_OversizeCounter(), 10),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )
    request = _pair(0)
    state = _state([request.top_page, request.bottom_page])
    with pytest.raises(InputBudgetExceeded, match='singleton'):
        node.dispatch_even(state)
    assert judge.calls == []
    assert rewriter.calls == []
