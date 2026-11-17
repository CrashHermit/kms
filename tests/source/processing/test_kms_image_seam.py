from __future__ import annotations

import asyncio

from PIL import Image

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.image_seam.models import ImageSeamDecision
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Judge:
    def __init__(self, continuation=True):
        self.continuation = continuation
        self.pairs = []

    async def acall(self, *, items):
        self.pairs.extend(items)
        return [
            ImageSeamDecision(
                position=item.position, is_continuation=self.continuation
            )
            for item in reversed(items)
        ]


def _node(judge: _Judge) -> ImageSeamNode:
    counter = MaterialTokenEstimator(_Counter())
    return ImageSeamNode(
        judge,
        budget=TokenBudget(counter=counter, token_limit=100_000),
        image_budget=ImagePayloadBudget(8, 26_214_400, 40_000_000),
        batch_size=8,
    )


def _fixture(tmp_path, page_indexes: list[int]):
    blocks = []
    images = {}
    for index in page_indexes:
        page = tmp_path / f'page-{index}.png'
        Image.new('RGB', (20, 20), 'white').save(page)
        images[index] = str(page)
        blocks.append(
            SourceBlock(
                uuid=f'image-{index}',
                block_type=BlockType.IMAGE,
                content=None,
                crop_path=str(page),
                crop_bbox=(2, 2, 10, 10),
                image_path=str(page),
            )
        )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        selected_page_indexes=page_indexes,
        ocr_page_images=images,
        ocr_block_pages={
            block.uuid: index
            for block, index in zip(blocks, page_indexes, strict=True)
        },
        text_seam_blocks=blocks,
    )
    return state, blocks


def test_four_consecutive_image_edges_fuse_once_from_physical_page_pairs(
    tmp_path,
):
    state, source_blocks = _fixture(tmp_path, [0, 1, 2, 3])
    node = _node(_Judge())

    sends = asyncio.run(node.dispatch(state))
    assert [
        (request.top_block.uuid, request.bottom_block.uuid)
        for send in sends
        for request in send.arg['image_seam_requests']
    ] == [
        ('image-0', 'image-1'),
        ('image-1', 'image-2'),
        ('image-2', 'image-3'),
    ]
    results = [
        result
        for send in sends
        for result in asyncio.run(node.worker(send.arg))['image_seam_results']
    ]
    state = state.model_copy(update={'image_seam_results': results})

    fused = asyncio.run(node.collect(state))['image_seam_blocks']

    assert len(fused) == 1
    assert fused[0].uuid == source_blocks[0].uuid
    assert fused[0].crop_bbox == source_blocks[0].crop_bbox
    assert fused[0].image_path != source_blocks[0].image_path


def test_selected_page_discontinuity_never_creates_an_image_edge(tmp_path):
    state, blocks = _fixture(tmp_path, [0, 2, 3])
    judge = _Judge()
    node = _node(judge)

    dispatch = asyncio.run(node.dispatch(state))

    assert [
        (request.top_block.uuid, request.bottom_block.uuid)
        for send in dispatch
        for request in send.arg['image_seam_requests']
    ] == [('image-2', 'image-3')]
    assert len(judge.pairs) == 0
    assert len(blocks) == 3


def test_reordered_page_selection_does_not_join_reversed_physical_edges(
    tmp_path,
):
    state, _ = _fixture(tmp_path, [1, 0])

    assert asyncio.run(_node(_Judge()).dispatch(state)) == 'image_seam_collect'


def test_text_merged_across_pages_does_not_hide_image_at_structural_anchor(
    tmp_path,
):
    state, images = _fixture(tmp_path, [0, 1, 2])
    merged_text = SourceBlock(
        uuid='merged-text',
        block_type=BlockType.PARAGRAPH,
        content='continued text',
    )
    state = state.model_copy(
        update={
            'text_seam_blocks': [merged_text, images[1], images[2]],
            'ocr_block_pages': {'merged-text': 0, 'image-1': 1, 'image-2': 2},
        }
    )
    node = _node(_Judge())

    sends = asyncio.run(node.dispatch(state))

    assert [
        (request.top_block.uuid, request.bottom_block.uuid)
        for send in sends
        for request in send.arg['image_seam_requests']
    ] == [('image-1', 'image-2')]


def test_rejected_edge_preserves_both_original_image_blocks(tmp_path):
    state, source_blocks = _fixture(tmp_path, [6, 7])
    node = _node(_Judge(continuation=False))

    sends = asyncio.run(node.dispatch(state))
    results = [
        result
        for send in sends
        for result in asyncio.run(node.worker(send.arg))['image_seam_results']
    ]
    state = state.model_copy(update={'image_seam_results': results})

    blocks = asyncio.run(node.collect(state))['image_seam_blocks']

    assert [block.uuid for block in blocks] == [
        block.uuid for block in source_blocks
    ]
