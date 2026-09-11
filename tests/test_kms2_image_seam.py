import asyncio

import dspy

from kms2.core.model import (
    BlockType,
    ImageSeamRequest,
    ImageSeamResult,
    Source,
    SourceBlock,
    SourcePage,
    VisualAsset,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source import image_seam
from kms2.node.source.image_seam import ImageSeamNode


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


def _state(pages: list[SourcePage]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        text_seam_pages=pages,
    )


class _Judge:
    def __init__(self, is_continuation: bool) -> None:
        self.is_continuation = is_continuation
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        return self.is_continuation


def test_signature_and_module_preserve_image_boundary(monkeypatch):
    assert set(image_seam.ImageSeamSignature.input_fields) == {
        'top_image',
        'bottom_image',
    }
    assert set(image_seam.ImageSeamSignature.output_fields) == {
        'is_continuation'
    }

    class FakeImage:
        def __init__(self, *, url: str) -> None:
            self.url = url

    monkeypatch.setattr(image_seam.dspy, 'Image', FakeImage)
    predictor = _Predictor(is_continuation=True)
    judge = image_seam.ImageSeamJudgeModule(predictor)
    top = _block(0, assets=[VisualAsset(path='top.png')])
    bottom = _block(1, assets=[VisualAsset(path='bottom.png')])

    assert judge(top_block=top, bottom_block=bottom) is True
    assert (
        asyncio.run(judge.aforward(top_block=top, bottom_block=bottom)) is True
    )
    assert [
        {key: value.url for key, value in call.items()}
        for call in predictor.calls
    ] == [
        {'top_image': 'top.png', 'bottom_image': 'bottom.png'},
        {'top_image': 'top.png', 'bottom_image': 'bottom.png'},
    ]


def test_edge_selection_skips_only_apparatus_and_rejects_invalid_images():
    judge = _Judge(False)
    node = ImageSeamNode(judge)
    eligible_top = _block(0, assets=[VisualAsset(path='top.png')])
    eligible_bottom = _block(1, assets=[VisualAsset(path='bottom.png')])
    top = SourcePage(
        index=0,
        blocks=[
            eligible_top,
            _block(2, 'citation', BlockType.BIBLIOGRAPHIC),
        ],
    )
    bottom = SourcePage(
        index=1,
        blocks=[
            _block(3, 'note', BlockType.NOTE),
            eligible_bottom,
        ],
    )

    asyncio.run(
        node.even_worker(
            {
                'image_seam_request': ImageSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )
    assert len(judge.calls) == 1
    assert judge.calls[0] == {
        'top_block': eligible_top,
        'bottom_block': eligible_bottom,
    }

    invalid_pairs = [
        (
            SourcePage(
                index=0,
                blocks=[eligible_top, _block(4, 'text', BlockType.TEXT)],
            ),
            bottom,
        ),
        (
            top,
            SourcePage(
                index=1,
                blocks=[_block(5, 'text', BlockType.TEXT), eligible_bottom],
            ),
        ),
        (
            SourcePage(index=0, blocks=[_block(6, assets=[])]),
            bottom,
        ),
        (
            SourcePage(
                index=0,
                blocks=[
                    _block(
                        7,
                        assets=[
                            VisualAsset(path='one.png'),
                            VisualAsset(path='two.png'),
                        ],
                    )
                ],
            ),
            bottom,
        ),
        (
            SourcePage(
                index=0,
                blocks=[
                    _block(
                        8, 'caption', assets=[VisualAsset(path='caption.png')]
                    )
                ],
            ),
            bottom,
        ),
    ]
    for invalid_top, invalid_bottom in invalid_pairs:
        assert node.dispatch_even(_state([invalid_top, invalid_bottom])) == (
            'image_seam_even_collect'
        )
    assert len(judge.calls) == 1


def test_true_merge_preserves_metadata_order_and_source_pages():
    top_context = _block(0, 'context', BlockType.TEXT)
    top_image = _block(1, assets=[VisualAsset(path='top.png')])
    bottom_image = _block(2, assets=[VisualAsset(path='bottom.png')])
    bottom_tail = _block(3, 'tail', BlockType.TEXT)
    top = SourcePage(index=0, blocks=[top_context, top_image])
    bottom = SourcePage(index=1, blocks=[bottom_image, bottom_tail])

    result = asyncio.run(
        ImageSeamNode(_Judge(True)).even_worker(
            {
                'image_seam_request': ImageSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )['image_seam_even_results'][0]

    assert [block.uuid for block in result.top_page.blocks] == [
        top_context.uuid,
        top_image.uuid,
    ]
    merged = result.top_page.blocks[1]
    assert [asset.path for asset in merged.assets] == ['top.png', 'bottom.png']
    assert merged.uuid == top_image.uuid
    assert merged.block_type is BlockType.IMAGE
    assert merged.content is None
    assert merged.crop_path == top_image.crop_path
    assert merged.crop_bbox == top_image.crop_bbox
    assert [block.uuid for block in result.bottom_page.blocks] == [
        bottom_tail.uuid
    ]
    assert [asset.path for asset in top_image.assets] == ['top.png']
    assert [asset.path for asset in bottom_image.assets] == ['bottom.png']


def test_false_judgment_preserves_both_pages():
    top = SourcePage(
        index=0, blocks=[_block(0, assets=[VisualAsset(path='top.png')])]
    )
    bottom = SourcePage(
        index=1, blocks=[_block(1, assets=[VisualAsset(path='bottom.png')])]
    )

    result = asyncio.run(
        ImageSeamNode(_Judge(False)).even_worker(
            {
                'image_seam_request': ImageSeamRequest(
                    top_page=top, bottom_page=bottom
                )
            }
        )
    )['image_seam_even_results'][0]

    assert result.top_page == top
    assert result.bottom_page == bottom


def test_dispatch_parity_and_collectors_preserve_order_and_empty_literal():
    pages = [
        SourcePage(
            index=index,
            blocks=[_block(index, assets=[VisualAsset(path=f'{index}.png')])],
        )
        for index in range(5)
    ]
    node = ImageSeamNode(_Judge(False))
    even_sends = node.dispatch_even(_state(pages))
    assert [
        (
            send.arg['image_seam_request'].top_page.index,
            send.arg['image_seam_request'].bottom_page.index,
        )
        for send in even_sends
    ] == [(0, 1), (2, 3)]
    assert node.dispatch_even(_state([pages[0]])) == 'image_seam_even_collect'

    state = _state(pages)
    state.image_seam_even_results = [
        ImageSeamResult(
            top_page=pages[2].model_copy(update={'blocks': []}),
            bottom_page=pages[3].model_copy(update={'blocks': []}),
        ),
        ImageSeamResult(
            top_page=pages[0].model_copy(update={'blocks': []}),
            bottom_page=pages[1].model_copy(update={'blocks': []}),
        ),
    ]
    collected = node.even_collect(state)['image_seam_even_pages']
    assert [page.index for page in collected] == [0, 1, 2, 3, 4]
    assert [len(page.blocks) for page in collected] == [0, 0, 0, 0, 1]
    assert [
        (
            send.arg['image_seam_request'].top_page.index,
            send.arg['image_seam_request'].bottom_page.index,
        )
        for send in node.dispatch_odd(
            state.model_copy(update={'image_seam_even_pages': pages})
        )
    ] == [(1, 2), (3, 4)]
    assert (
        node.dispatch_odd(
            state.model_copy(update={'image_seam_even_pages': [pages[0]]})
        )
        == 'image_seam_odd_collect'
    )

    odd_state = state.model_copy(update={'image_seam_even_pages': pages})
    odd_state.image_seam_odd_results = [
        ImageSeamResult(
            top_page=pages[3].model_copy(update={'blocks': []}),
            bottom_page=pages[4].model_copy(update={'blocks': []}),
        )
    ]
    odd_collected = node.odd_collect(odd_state)['image_seam_pages']
    assert [len(page.blocks) for page in odd_collected] == [1, 1, 1, 0, 0]
