import asyncio
import base64
from pathlib import Path

import dspy
from PIL import Image

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.content_correction import (
    ContentCorrectionInput,
    ContentCorrectionOutput,
    ContentCorrectionRequest,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms2.node.source_processing.content_correction import ContentCorrectionNode


class _Counter:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.texts.extend(texts)
        return [1] * len(texts)


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[list[ContentCorrectionInput], list[dspy.Image]]
        ] = []
        self.output_order: list[int] | None = None

    def __call__(
        self,
        *,
        blocks: list[ContentCorrectionInput],
        block_crops: list[dspy.Image],
    ) -> dspy.Prediction:
        self.calls.append((blocks, block_crops))
        outputs = [
            ContentCorrectionOutput(
                position=block.position,
                corrected_content=(
                    None
                    if block.content is None
                    else f'corrected {block.content}'
                ),
            )
            for block in blocks
        ]
        if self.output_order is not None:
            by_position = {output.position: output for output in outputs}
            outputs = [by_position[position] for position in self.output_order]
        return dspy.Prediction(corrected_blocks=outputs)

    async def acall(
        self,
        *,
        blocks: list[ContentCorrectionInput],
        block_crops: list[dspy.Image],
    ) -> dspy.Prediction:
        return self(blocks=blocks, block_crops=block_crops)


def _image(path: Path, size: tuple[int, int], color: str = 'red') -> None:
    Image.new('RGB', size, color=color).save(path)


def _node(
    predictor: _Predictor,
    *,
    token_limit: int = 10_000,
    batch_size: int = 32,
    counter: _Counter | None = None,
) -> ContentCorrectionNode:
    return ContentCorrectionNode(
        ContentCorrectorModule(predictor),
        budget=TokenBudget(
            counter=counter or _Counter(), token_limit=token_limit
        ),
        batch_size=batch_size,
    )


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(uuid='source-1', key='document.pdf'),
        ocr_pages=pages,
    )


def _block(
    uuid: str,
    crop_path: Path,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    content: str | None = 'transcription',
    assets: list[dict[str, str]] | None = None,
) -> SourceBlock:
    return SourceBlock(
        uuid=uuid,
        block_type=block_type,
        content=content,
        crop_path=str(crop_path),
        assets=assets or [],
    )


def test_signature_formats_crops_as_ordered_native_image_blocks():
    first = dspy.Image(url='data:image/png;base64,AAAA')
    second = dspy.Image(url='data:image/png;base64,BBBB')
    blocks = [
        ContentCorrectionInput(
            position=0,
            block_type=BlockType.PARAGRAPH,
            content='first',
        ),
        ContentCorrectionInput(
            position=1,
            block_type=BlockType.IMAGE,
            content=None,
        ),
    ]

    messages = dspy.ChatAdapter().format(
        ContentCorrectorSignature,
        demos=[],
        inputs={'blocks': blocks, 'block_crops': [first, second]},
    )

    image_blocks = [
        block
        for block in messages[1]['content']
        if block['type'] == 'image_url'
    ]
    assert [block['image_url']['url'] for block in image_blocks] == [
        first.url,
        second.url,
    ]


def test_worker_batches_across_pages_and_preserves_request_ownership(tmp_path):
    crop_paths = [tmp_path / f'{index}.png' for index in range(3)]
    for path, color in zip(crop_paths, ['red', 'green', 'blue'], strict=True):
        _image(path, (2, 2), color)
    first = _block(
        'first',
        crop_paths[0],
        assets=[{'path': 'first-asset.png'}],
    )
    empty_text = _block('empty-text', crop_paths[1], content=None)
    image_block = _block(
        'image',
        crop_paths[2],
        block_type=BlockType.IMAGE,
        content='image label',
        assets=[{'path': 'image-asset.png'}],
    )
    state = _state(
        [
            SourcePage(index=4, blocks=[first, empty_text]),
            SourcePage(index=9, blocks=[image_block]),
        ]
    )
    predictor = _Predictor()
    predictor.output_order = [2, 0, 1]
    node = _node(predictor)

    [send] = node.dispatch(state)
    requests = send.arg['content_correction_requests']
    assert requests == [
        ContentCorrectionRequest(
            page_index=4, block_position=0, source_block=first
        ),
        ContentCorrectionRequest(
            page_index=4, block_position=1, source_block=empty_text
        ),
        ContentCorrectionRequest(
            page_index=9, block_position=0, source_block=image_block
        ),
    ]
    worker_result = asyncio.run(node.worker(send.arg))
    state.correction_results = worker_result['correction_results']
    assert len(predictor.calls) == 1
    assert [
        base64.b64decode(image.url.split(',', 1)[1])
        for image in predictor.calls[0][1]
    ] == [path.read_bytes() for path in crop_paths]

    corrected_pages = node.collect(state)['corrected_pages']
    assert [block.content for block in corrected_pages[0].blocks] == [
        'corrected transcription',
        None,
    ]
    assert corrected_pages[1].blocks[0].content == 'corrected image label'
    assert corrected_pages[0].blocks[0].uuid == first.uuid
    assert corrected_pages[0].blocks[0].assets == first.assets
    assert corrected_pages[0].blocks[1].uuid == empty_text.uuid
    assert corrected_pages[1].blocks[0].uuid == image_block.uuid
    assert corrected_pages[1].blocks[0].assets == image_block.assets


def test_batch_size_one_sends_each_source_block_in_order(tmp_path):
    paths = [tmp_path / f'{index}.png' for index in range(3)]
    for path in paths:
        _image(path, (1, 1))
    blocks = [
        _block(f'block-{index}', path, content=f'text-{index}')
        for index, path in enumerate(paths)
    ]
    node = _node(_Predictor(), batch_size=1)

    sends = node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))

    assert [
        [
            request.source_block.uuid
            for request in send.arg['content_correction_requests']
        ]
        for send in sends
    ] == [['block-0'], ['block-1'], ['block-2']]


def test_dispatch_splits_by_text_budget_and_preserves_source_order(tmp_path):
    blocks = [
        _block(
            f'block-{index}', tmp_path / f'{index}.png', content=f'text-{index}'
        )
        for index in range(4)
    ]
    node = _node(_Predictor(), batch_size=4, token_limit=4)

    sends = node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))

    assert [len(send.arg['content_correction_requests']) for send in sends] == [
        2,
        2,
    ]
    assert [
        request.source_block.uuid
        for send in sends
        for request in send.arg['content_correction_requests']
    ] == [block.uuid for block in blocks]
