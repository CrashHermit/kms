from __future__ import annotations

import asyncio

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.text_seam.models import (
    TextSeamDecision,
    TextSeamRewriteOutput,
)
from kms.source.processing.text_seam.node import TextSeamNode


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Judge:
    def __init__(self, rejected_tail: str | None = None) -> None:
        self.rejected_tail = rejected_tail
        self.seen = []

    async def acall(self, *, items):
        self.seen.extend(items)
        return [
            TextSeamDecision(
                position=item.position,
                is_split=item.tail.material.text != self.rejected_tail,
            )
            for item in reversed(items)
        ]


class _Rewriter:
    def __init__(self) -> None:
        self.items = []

    async def acall(self, *, items):
        self.items.extend(items)
        return [
            TextSeamRewriteOutput(
                position=item.position,
                merged=''.join(block.material.text for block in item.blocks),
            )
            for item in reversed(items)
        ]


def _node(judge: _Judge, rewriter: _Rewriter) -> TextSeamNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    return TextSeamNode(
        judge,
        rewriter,
        judge_budget=budget,
        rewrite_budget=budget,
        judge_batch_size=16,
        rewrite_batch_size=16,
    )


def _state(page_indexes: list[int], *, empty_pages=()) -> SourceProcessingState:
    blocks = [
        SourceBlock(
            uuid=f'b{index}',
            block_type=BlockType.PARAGRAPH,
            content=f'fragment-{index}',
            crop_path=f'crop-{index}.png',
            crop_bbox=(index, 0, index + 1, 1),
            embedding=[float(index)],
        )
        for index in page_indexes
        if index not in empty_pages
    ]
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        selected_page_indexes=page_indexes,
        ocr_block_pages={block.uuid: int(block.uuid[1:]) for block in blocks},
        formatted_blocks=blocks,
    )


def test_four_page_continuation_forms_one_chain_and_rewrites_once():
    state = _state([1, 2, 3, 4])
    judge = _Judge()
    rewriter = _Rewriter()
    node = _node(judge, rewriter)

    judge_sends = asyncio.run(node.dispatch(state))
    judge_results = [
        result
        for send in judge_sends
        for result in asyncio.run(node.worker(send.arg))['text_seam_results']
    ]
    state = state.model_copy(update={'text_seam_results': judge_results})
    requests = node.prepare_rewrites(state)['text_seam_rewrite_requests']
    assert len(requests) == 1
    assert [block.uuid for block in requests[0].blocks] == [
        'b1',
        'b2',
        'b3',
        'b4',
    ]

    state = state.model_copy(update={'text_seam_rewrite_requests': requests})
    rewrite_sends = asyncio.run(node.dispatch_rewrites(state))
    rewrite_results = [
        result
        for send in rewrite_sends
        for result in asyncio.run(node.rewrite_worker(send.arg))[
            'text_seam_rewrite_results'
        ]
    ]
    state = state.model_copy(
        update={'text_seam_rewrite_results': rewrite_results}
    )
    blocks = node.collect(state)['text_seam_blocks']

    assert len(judge.seen) == 3
    assert len(rewriter.items) == 1
    assert len(rewriter.items[0].blocks) == 4
    assert len(blocks) == 1
    assert blocks[0].uuid == 'b1'
    assert blocks[0].content == 'fragment-1fragment-2fragment-3fragment-4'
    assert blocks[0].crop_path == 'crop-1.png'
    assert blocks[0].embedding is None


def test_rejected_edge_remains_a_boundary_and_other_chain_still_merges():
    state = _state([0, 1, 2])
    judge = _Judge(rejected_tail='fragment-0')
    node = _node(judge, _Rewriter())

    sends = asyncio.run(node.dispatch(state))
    results = [
        result
        for send in sends
        for result in asyncio.run(node.worker(send.arg))['text_seam_results']
    ]
    state = state.model_copy(update={'text_seam_results': results})
    requests = node.prepare_rewrites(state)['text_seam_rewrite_requests']

    assert len(requests) == 1
    assert [block.uuid for block in requests[0].blocks] == ['b1', 'b2']

    state = state.model_copy(update={'text_seam_rewrite_requests': requests})
    rewrite_sends = asyncio.run(node.dispatch_rewrites(state))
    state = state.model_copy(
        update={
            'text_seam_rewrite_results': [
                result
                for send in rewrite_sends
                for result in asyncio.run(node.rewrite_worker(send.arg))[
                    'text_seam_rewrite_results'
                ]
            ]
        }
    )
    blocks = node.collect(state)['text_seam_blocks']

    assert [block.uuid for block in blocks] == ['b0', 'b1']
    assert blocks[0].content == 'fragment-0'
    assert blocks[1].content == 'fragment-1fragment-2'


def test_empty_omitted_or_reordered_pages_do_not_create_physical_seams():
    judge = _Judge()
    rewriter = _Rewriter()
    node = _node(judge, rewriter)

    for state in (
        _state([0, 2]),
        _state([0, 1, 2], empty_pages={1}),
        _state([1, 0]),
    ):
        assert asyncio.run(node.dispatch(state)) == 'text_seam_collect'

    assert judge.seen == []
    assert rewriter.items == []
