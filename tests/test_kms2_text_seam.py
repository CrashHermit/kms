import asyncio

import dspy

from kms2.core.model import (
    BlockType,
    Source,
    SourceBlock,
    SourcePage,
    VisualAsset,
)
from kms2.core.model.source_stage.text_seam import (
    TextSeamRequest,
    TextSeamResult,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source import text_seam
from kms2.node.source.text_seam import TextSeamNode


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


def _module(
    module_type: type[text_seam.TextSeamJudgeModule]
    | type[text_seam.TextSeamRewriterModule],
    predictor: _Predictor,
):
    return module_type(predictor)


def _state(pages: list[SourcePage]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        formatted_pages=pages,
    )


def test_signatures_and_module_calls_preserve_exact_text_boundary():
    assert set(text_seam.TextSeamSignature.input_fields) == {
        'top_node_context',
        'top_bottom_edge_node',
        'bottom_top_edge_node',
        'bottom_node_context',
    }
    assert set(text_seam.TextSeamSignature.output_fields) == {'is_split'}
    assert set(text_seam.TextSeamRewriteSignature.input_fields) == {
        'tail',
        'head',
        'tail_kind',
        'head_kind',
        'before_tail',
        'after_head',
    }
    assert set(text_seam.TextSeamRewriteSignature.output_fields) == {'merged'}

    tail = _block(1, 'tail')
    head = _block(2, 'head', BlockType.PARAGRAPH)
    judge_predictor = _Predictor(is_split=True)
    judge = _module(text_seam.TextSeamJudgeModule, judge_predictor)
    assert (
        judge(
            tail=tail,
            head=head,
            top_context=None,
            bottom_context=None,
        )
        is True
    )
    assert judge_predictor.calls == [
        {
            'top_node_context': '',
            'top_bottom_edge_node': 'tail',
            'bottom_top_edge_node': 'head',
            'bottom_node_context': '',
        }
    ]
    assert (
        asyncio.run(
            judge.aforward(
                tail=tail, head=head, top_context=None, bottom_context=None
            )
        )
        is True
    )

    rewriter_predictor = _Predictor(merged='tail head')
    rewriter = _module(text_seam.TextSeamRewriterModule, rewriter_predictor)
    assert (
        rewriter(
            tail=tail,
            head=head,
            top_context=None,
            bottom_context=None,
        )
        == 'tail head'
    )
    assert rewriter_predictor.calls == [
        {
            'tail': 'tail',
            'head': 'head',
            'tail_kind': 'paragraph',
            'head_kind': 'paragraph',
            'before_tail': '',
            'after_head': '',
        }
    ]
    assert (
        asyncio.run(
            rewriter.aforward(
                tail=tail,
                head=head,
                top_context=None,
                bottom_context=None,
            )
        )
        == 'tail head'
    )


class _Judge:
    def __init__(self, is_split: bool) -> None:
        self.is_split = is_split
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        return self.is_split


class _Rewriter:
    def __init__(self, merged: str = 'merged') -> None:
        self.merged = merged
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.merged


def test_true_merge_preserves_tail_metadata_and_removes_head():
    top_context = _block(0, 'before')
    tail = _block(1, 'tail')
    head = _block(4, 'head')
    bottom_context = _block(5, 'after')
    top = SourcePage(index=0, blocks=[top_context, tail])
    bottom = SourcePage(index=1, blocks=[head, bottom_context])
    judge = _Judge(True)
    rewriter = _Rewriter('merged text')

    result = asyncio.run(
        TextSeamNode(judge, rewriter).even_worker(
            {
                'text_seam_request': TextSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )['text_seam_even_results'][0]

    assert [block.content for block in result.top_page.blocks] == [
        'before',
        'merged text',
    ]
    assert [block.content for block in result.bottom_page.blocks] == ['after']
    merged_tail = result.top_page.blocks[1]
    assert merged_tail.uuid == tail.uuid
    assert merged_tail.block_type is tail.block_type
    assert merged_tail.assets == tail.assets
    assert merged_tail.crop_path == tail.crop_path
    assert merged_tail.crop_bbox == tail.crop_bbox
    assert judge.calls[0]['top_context'] == top_context
    assert judge.calls[0]['bottom_context'] == bottom_context
    assert len(rewriter.calls) == 1
    assert top.blocks[1].content == 'tail'
    assert bottom.blocks[0].content == 'head'


def test_false_judgment_preserves_pages_and_skips_rewriter():
    top = SourcePage(index=0, blocks=[_block(0, 'tail')])
    bottom = SourcePage(index=1, blocks=[_block(0, 'head')])
    judge = _Judge(False)
    rewriter = _Rewriter()

    result = asyncio.run(
        TextSeamNode(judge, rewriter).even_worker(
            {
                'text_seam_request': TextSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )['text_seam_even_results'][0]

    assert result.top_page == top
    assert result.bottom_page == bottom
    assert rewriter.calls == []


def test_selection_skips_visual_apparatus_and_asset_blocks_for_edges_and_context():
    asset_block = _block(2, 'asset', assets=[VisualAsset(path='asset.png')])
    top = SourcePage(
        index=0,
        blocks=[
            _block(0, 'top context'),
            _block(1, 'bibliographic', BlockType.BIBLIOGRAPHIC),
            _block(2, 'image', BlockType.IMAGE),
            asset_block,
            _block(3, 'tail'),
        ],
    )
    bottom = SourcePage(
        index=1,
        blocks=[
            _block(4, 'head'),
            _block(5, 'image', BlockType.IMAGE),
            _block(6, 'note', BlockType.NOTE),
            _block(7, 'asset after', assets=[VisualAsset(path='after.png')]),
            _block(8, 'bottom context'),
        ],
    )
    judge = _Judge(False)

    asyncio.run(
        TextSeamNode(judge, _Rewriter()).even_worker(
            {
                'text_seam_request': TextSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )

    assert judge.calls[0]['tail'].content == 'tail'
    assert judge.calls[0]['head'].content == 'head'
    assert judge.calls[0]['top_context'].content == 'top context'
    assert judge.calls[0]['bottom_context'].content == 'bottom context'

    invalid_top = SourcePage(index=0, blocks=[_block(0, ''), _block(1, 'tail')])
    invalid_bottom = SourcePage(
        index=1, blocks=[_block(0, ''), _block(1, 'head')]
    )
    node = TextSeamNode(_Judge(True), _Rewriter())
    assert node.dispatch_even(_state([invalid_top, invalid_bottom])) == (
        'text_seam_even_collect'
    )


def test_dispatch_uses_upper_page_parity_and_collectors_preserve_page_order():
    pages = [
        SourcePage(index=index, blocks=[_block(index, f'page {index}')])
        for index in range(5)
    ]
    node = TextSeamNode(_Judge(False), _Rewriter())

    even_sends = node.dispatch_even(_state(pages))
    assert [
        (
            send.arg['text_seam_request'].top_page.index,
            send.arg['text_seam_request'].bottom_page.index,
        )
        for send in even_sends
    ] == [
        (0, 1),
        (2, 3),
    ]
    even_state = _state(pages)
    even_state.text_seam_even_results = [
        TextSeamResult(
            top_page=pages[2].model_copy(
                update={'blocks': [_block(2, 'merged 2')]}
            ),
            bottom_page=pages[3].model_copy(update={'blocks': []}),
        ),
        TextSeamResult(
            top_page=pages[0].model_copy(
                update={'blocks': [_block(0, 'merged 0')]}
            ),
            bottom_page=pages[1].model_copy(update={'blocks': []}),
        ),
    ]
    collected = node.even_collect(even_state)['text_seam_even_pages']
    assert [page.index for page in collected] == [0, 1, 2, 3, 4]
    assert [
        page.blocks[0].content if page.blocks else None for page in collected
    ] == [
        'merged 0',
        None,
        'merged 2',
        None,
        'page 4',
    ]
    odd_sends = node.dispatch_odd(
        even_state.model_copy(update={'text_seam_even_pages': pages})
    )
    assert [
        (
            send.arg['text_seam_request'].top_page.index,
            send.arg['text_seam_request'].bottom_page.index,
        )
        for send in odd_sends
    ] == [(1, 2), (3, 4)]
    assert node.dispatch_odd(_state([SourcePage(index=0, blocks=[])])) == (
        'text_seam_odd_collect'
    )
