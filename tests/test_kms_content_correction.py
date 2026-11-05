import asyncio
import base64
from io import BytesIO
from pathlib import Path

import dspy
import pytest
from PIL import Image

from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.content_correction import (
    ContentCorrectionBlockContext,
    ContentCorrectionInput,
    ContentCorrectionOutput,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.module.source_processing.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms.node.source_processing.content_correction import ContentCorrectionNode


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
        content_images: list[dspy.Image],
    ) -> dspy.Prediction:
        self.calls.append((blocks, content_images))
        outputs = [
            ContentCorrectionOutput(
                position=block.position,
                keep=block.material.text != 'reject',
                corrected_content=(
                    None
                    if block.material.text is None
                    or block.material.text == 'reject'
                    else f'corrected {block.material.text}'
                ),
                cleanup_instructions=(
                    ['Remove the identified footer.']
                    if block.material.text == 'mixed'
                    else []
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
        content_images: list[dspy.Image],
    ) -> dspy.Prediction:
        return self(blocks=blocks, content_images=content_images)


def _image(path: Path, size: tuple[int, int], color: str = 'red') -> None:
    Image.new('RGB', size, color=color).save(path)


def _node(
    predictor: _Predictor,
    *,
    token_limit: int = 10_000,
    batch_size: int = 32,
    counter: _Counter | None = None,
    max_pixels: int = 10_000_000,
) -> ContentCorrectionNode:
    return ContentCorrectionNode(
        ContentCorrectorModule(predictor),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(counter or _Counter()),
            token_limit=token_limit,
        ),
        image_budget=ImagePayloadBudget(
            max_images=32, max_bytes=10_000_000, max_pixels=max_pixels
        ),
        batch_size=batch_size,
    )


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    directory = next(
        Path(block.crop_path).parent
        for page in pages
        for block in page.blocks
        if block.crop_path is not None
    )
    page_images = {}
    for page in pages:
        path = directory / f'fixture-page-{page.index}.png'
        _image(path, (32, 32), 'white')
        page_images[page.index] = str(path)
    return SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'document.pdf'},
        source=Source(uuid='source-1', key='document.pdf'),
        ocr_pages=pages,
        ocr_page_images=page_images,
    )


def _block(
    uuid: str,
    crop_path: Path,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    content: str | None = 'transcription',
    assets: list[dict[str, str]] | None = None,
    crop_bbox: tuple[int, int, int, int] = (0, 0, 1, 1),
) -> SourceBlock:
    return SourceBlock(
        uuid=uuid,
        block_type=block_type,
        content=content,
        crop_path=str(crop_path),
        crop_bbox=crop_bbox,
        assets=assets or [],
    )


def test_signature_formats_crops_as_ordered_native_image_blocks():
    first = dspy.Image(url='data:image/png;base64,AAAA')
    blocks = [
        ContentCorrectionInput(
            position=0,
            block_type=BlockType.PARAGRAPH,
            material=MultimodalContent(text='text', image_positions=[0]),
            context=ContentCorrectionBlockContext(
                image_position=0, target_bounds=(0, 0, 1, 1)
            ),
        ),
    ]

    messages = dspy.ChatAdapter().format(
        ContentCorrectorSignature,
        demos=[],
        inputs={'blocks': blocks, 'content_images': [first]},
    )
    image_blocks = [
        block
        for block in messages[1]['content']
        if block['type'] == 'image_url'
    ]
    assert [block['image_url']['url'] for block in image_blocks] == [first.url]


def test_worker_maps_reordered_outputs_and_collects_rejections_by_original_location(
    tmp_path,
):
    crop_paths = [tmp_path / f'{index}.png' for index in range(4)]
    for path, color in zip(
        crop_paths, ['red', 'green', 'blue', 'yellow'], strict=True
    ):
        _image(path, (2, 2), color)
    mixed = _block(
        'mixed',
        crop_paths[0],
        content='mixed',
        assets=[{'path': 'mixed-asset.png'}],
    )
    rejected = _block('rejected', crop_paths[1], content='reject')
    blank = _block('blank', crop_paths[2], content=None)
    image_block = _block(
        'image',
        crop_paths[3],
        block_type=BlockType.IMAGE,
        content='image label',
        assets=[{'path': 'image-asset.png'}],
    )
    state = _state(
        [
            SourcePage(index=4, markdown='raw first', blocks=[mixed, rejected]),
            SourcePage(
                index=9, markdown='raw second', blocks=[blank, image_block]
            ),
            SourcePage(index=12, markdown='raw empty', blocks=[]),
        ]
    )
    predictor = _Predictor()
    predictor.output_order = [2, 0, 1]
    node = _node(predictor)

    [send] = asyncio.run(node.dispatch(state))
    requests = send.arg['content_correction_requests']
    assert [
        (request.page_index, request.block_position, request.source_block)
        for request in requests
    ] == [(4, 0, mixed), (4, 1, rejected), (9, 0, blank)]
    worker_result = asyncio.run(node.worker(send.arg))
    state.correction_results = worker_result['correction_results']
    assert len(predictor.calls) == 1

    corrected_pages = node.collect(state)['corrected_pages']
    assert [block.content for block in corrected_pages[0].blocks] == [
        'corrected mixed',
    ]
    assert corrected_pages[1].blocks == [blank, image_block]
    assert corrected_pages[2].blocks == []
    assert [page.index for page in corrected_pages] == [4, 9, 12]
    assert [page.markdown for page in corrected_pages] == [
        'raw first',
        'raw second',
        'raw empty',
    ]
    assert corrected_pages[0].blocks[0].uuid == mixed.uuid
    assert corrected_pages[0].blocks[0].assets == mixed.assets
    assert corrected_pages[1].blocks[0].uuid == blank.uuid
    assert corrected_pages[1].blocks[1].uuid == image_block.uuid
    assert corrected_pages[1].blocks[1].assets == image_block.assets
    assert [
        result.cleanup_instructions for result in state.correction_results
    ] == [['Remove the identified footer.'], [], []]


def test_shared_crop_path_deduplicates_without_changing_block_positions(
    tmp_path,
):
    crop_path = tmp_path / 'shared.png'
    _image(crop_path, (2, 2))
    blocks = [
        _block('first', crop_path, content='first'),
        _block('second', crop_path, content='second'),
    ]
    predictor = _Predictor()
    node = _node(predictor)

    [send] = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))
    )
    result = asyncio.run(node.worker(send.arg))
    assert [
        (item.page_index, item.block_position)
        for item in result['correction_results']
    ] == [(0, 0), (0, 1)]

    inputs, content_images = predictor.calls[0]
    assert [block.position for block in inputs] == [0, 1]
    assert [block.material.image_positions for block in inputs] == [[0], [0]]
    assert [block.context.image_position for block in inputs] == [1, 1]
    assert len(content_images) == 2


def test_batch_size_one_sends_each_source_block_in_order(tmp_path):
    paths = [tmp_path / f'{index}.png' for index in range(3)]
    for path in paths:
        _image(path, (1, 1))
    blocks = [
        _block(f'block-{index}', path, content=f'text-{index}')
        for index, path in enumerate(paths)
    ]
    node = _node(_Predictor(), batch_size=1)

    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))
    )
    assert [
        [
            request.source_block.uuid
            for request in send.arg['content_correction_requests']
        ]
        for send in sends
    ] == [['block-0'], ['block-1'], ['block-2']]


def test_dispatch_splits_by_combined_budget_and_preserves_source_order(
    tmp_path,
):
    paths = [tmp_path / f'{index}.png' for index in range(4)]
    for path in paths:
        _image(path, (2, 2))
    blocks = [
        _block(f'block-{index}', path, content=f'text-{index}')
        for index, path in enumerate(paths)
    ]
    node = _node(_Predictor(), batch_size=4, token_limit=5)

    sends = asyncio.run(
        node.dispatch(_state([SourcePage(index=0, blocks=blocks)]))
    )
    assert [len(send.arg['content_correction_requests']) for send in sends] == [
        1,
        1,
        1,
        1,
    ]
    assert [
        request.source_block.uuid
        for send in sends
        for request in send.arg['content_correction_requests']
    ] == [block.uuid for block in blocks]


def _payload(image: dspy.Image) -> bytes:
    return base64.b64decode(image.url.split(',', 1)[1])


def _paint(
    path: Path,
    size: tuple[int, int],
    blocks: list[tuple[tuple[int, int, int, int], str]],
) -> None:
    image = Image.new('RGB', size, 'white')
    for bounds, color in blocks:
        image.paste(
            Image.new(
                'RGB',
                (bounds[2] - bounds[0], bounds[3] - bounds[1]),
                color,
            ),
            bounds[:2],
        )
    image.save(path)


def test_positioned_context_uses_previous_target_next_blocks(tmp_path):
    page = tmp_path / 'page.png'
    _paint(
        page,
        (16, 12),
        [
            ((1, 1, 4, 3), 'red'),
            ((6, 4, 10, 6), 'green'),
            ((11, 8, 15, 11), 'blue'),
        ],
    )
    crop = tmp_path / 'target-crop.png'
    _image(crop, (2, 1), 'yellow')
    blocks = [
        _block('before', crop, crop_bbox=(1, 1, 4, 3)),
        _block('target', crop, crop_bbox=(6, 4, 10, 6)),
        _block('after', crop, crop_bbox=(11, 8, 15, 11)),
        _block('outside-window', crop, crop_bbox=(0, 10, 1, 11)),
    ]
    state = _state([SourcePage(index=0, blocks=blocks)])
    state.ocr_page_images = {0: str(page)}
    originals = (page.read_bytes(), crop.read_bytes())
    predictor = _Predictor()
    node = _node(predictor)

    [send] = asyncio.run(node.dispatch(state))
    request = send.arg['content_correction_requests'][1]
    assert request.source_block.uuid == 'target'
    state.correction_results = asyncio.run(node.worker(send.arg))[
        'correction_results'
    ]

    inputs, images = predictor.calls[0]
    assert [item.position for item in inputs] == [0, 1, 2, 3]
    assert inputs[1].context.target_bounds == (5, 3, 9, 5)
    assert len(images) == 5
    with Image.open(BytesIO(_payload(images[2]))) as canvas:
        assert canvas.size == (14, 10)
        assert canvas.getpixel((2, 1)) == (255, 0, 0)
        assert canvas.getpixel((7, 4)) == (0, 128, 0)
        assert canvas.getpixel((12, 9)) == (0, 0, 255)
    assert _payload(images[0]) == crop.read_bytes()
    assert (page.read_bytes(), crop.read_bytes()) == originals
    results = node.collect(state)['corrected_pages']
    assert [block.uuid for block in results[0].blocks] == [
        'before',
        'target',
        'after',
        'outside-window',
    ]


def test_context_crosses_adjacent_pages_and_uses_canvas_coordinates(tmp_path):
    pages = {}
    for index, color in ((4, 'red'), (5, 'green'), (6, 'blue')):
        path = tmp_path / f'page-{index}.png'
        _paint(path, (8, 4), [((2, 1, 5, 3), color)])
        pages[index] = str(path)
    crop = tmp_path / 'crop.png'
    _image(crop, (1, 1), 'yellow')
    source_pages = [
        SourcePage(
            index=index,
            blocks=[
                _block(
                    f'block-{index}',
                    crop,
                    crop_bbox=(2, 1, 5, 3),
                )
            ],
        )
        for index in (4, 5, 6)
    ]
    state = _state(source_pages)
    state.ocr_page_images = pages
    predictor = _Predictor()
    node = _node(predictor, batch_size=1)

    sends = asyncio.run(node.dispatch(state))
    assert [
        send.arg['content_correction_requests'][0].page_index for send in sends
    ] == [4, 5, 6]
    asyncio.run(node.worker(sends[1].arg))
    inputs, images = predictor.calls[0]
    assert inputs[0].context.target_bounds == (0, 4, 3, 6)
    with Image.open(BytesIO(_payload(images[1]))) as canvas:
        assert canvas.size == (3, 10)
        assert canvas.getpixel((1, 1)) == (255, 0, 0)
        assert canvas.getpixel((1, 5)) == (0, 128, 0)
        assert canvas.getpixel((1, 8)) == (0, 0, 255)


@pytest.mark.parametrize(
    ('target_page', 'expected_bounds', 'expected_color'),
    [
        (0, (0, 0, 2, 2), (255, 0, 0)),
        (2, (0, 4, 2, 6), (0, 0, 255)),
    ],
)
def test_first_and_last_blocks_keep_correct_context_positions(
    tmp_path, target_page, expected_bounds, expected_color
):
    crop = tmp_path / 'crop.png'
    _image(crop, (1, 1))
    pages = []
    page_images = {}
    for index, color in enumerate(('red', 'green', 'blue')):
        path = tmp_path / f'page-{index}.png'
        _paint(path, (6, 4), [((1, 1, 3, 3), color)])
        page_images[index] = str(path)
        pages.append(
            SourcePage(
                index=index,
                blocks=[
                    _block(
                        f'block-{index}',
                        crop,
                        crop_bbox=(1, 1, 3, 3),
                    )
                ],
            )
        )
    state = _state(pages)
    state.ocr_page_images = page_images
    predictor = _Predictor()
    node = _node(predictor, batch_size=1)

    sends = asyncio.run(node.dispatch(state))
    asyncio.run(node.worker(sends[target_page].arg))
    inputs, images = predictor.calls[0]
    assert inputs[0].context.target_bounds == expected_bounds
    with Image.open(BytesIO(_payload(images[1]))) as canvas:
        assert canvas.getpixel((1, expected_bounds[1])) == expected_color


def test_nonconsecutive_pages_do_not_enter_context_canvas(tmp_path):
    crop = tmp_path / 'crop.png'
    _image(crop, (1, 1))
    pages = []
    page_images = {}
    for index, color in ((9, 'red'), (10, 'green'), (12, 'blue')):
        path = tmp_path / f'page-{index}.png'
        _paint(path, (6, 4), [((1, 1, 3, 3), color)])
        page_images[index] = str(path)
        pages.append(
            SourcePage(
                index=index,
                blocks=[
                    _block(
                        f'block-{index}',
                        crop,
                        crop_bbox=(1, 1, 3, 3),
                    )
                ],
            )
        )
    state = _state(pages)
    state.ocr_page_images = page_images
    predictor = _Predictor()
    node = _node(predictor, batch_size=1)

    sends = asyncio.run(node.dispatch(state))
    asyncio.run(node.worker(sends[1].arg))
    inputs, images = predictor.calls[0]
    assert inputs[0].context.target_bounds == (0, 4, 2, 6)
    with Image.open(BytesIO(_payload(images[1]))) as canvas:
        assert canvas.size == (2, 6)
        assert canvas.getpixel((1, 0)) == (255, 0, 0)
        assert canvas.getpixel((1, 4)) == (0, 128, 0)


def test_context_canvas_pixel_budget_is_enforced(tmp_path):
    page = tmp_path / 'page.png'
    _image(page, (4, 4), 'red')
    crop = tmp_path / 'crop.png'
    _image(crop, (1, 1))
    block = _block('target', crop, crop_bbox=(0, 0, 1, 1))
    state = _state([SourcePage(index=0, blocks=[block])])
    state.ocr_page_images = {0: str(page)}
    node = _node(_Predictor(), max_pixels=1)

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(node.dispatch(state))


def test_target_crop_and_context_canvas_count_toward_token_budget(tmp_path):
    page = tmp_path / 'page.png'
    _image(page, (1, 1))
    crop = tmp_path / 'crop.png'
    _image(crop, (1, 1))
    state = _state(
        [
            SourcePage(
                index=0,
                blocks=[
                    _block(
                        'target',
                        crop,
                        crop_bbox=(0, 0, 1, 1),
                    )
                ],
            )
        ]
    )
    state.ocr_page_images = {0: str(page)}
    node = _node(_Predictor(), token_limit=3)

    with pytest.raises(
        InputBudgetExceeded, match='Content exceeds the configured input budget'
    ):
        asyncio.run(node.dispatch(state))


def test_all_image_input_bypasses_corrector_and_preserves_page(tmp_path):
    crop = tmp_path / 'image.png'
    _image(crop, (1, 1))
    image_block = _block('image', crop, block_type=BlockType.IMAGE)
    state = _state([SourcePage(index=4, blocks=[image_block])])
    predictor = _Predictor()
    node = _node(predictor)

    assert asyncio.run(node.dispatch(state)) == 'content_correction_collect'
    assert node.collect(state)['corrected_pages'] == state.ocr_pages
    assert predictor.calls == []
