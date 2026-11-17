from __future__ import annotations

from PIL import Image

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.correction.models import ContentCorrectionResult
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _block(uuid: str, content: str, *, kind=BlockType.PARAGRAPH) -> SourceBlock:
    return SourceBlock(
        uuid=uuid,
        block_type=kind,
        content=content,
        crop_path=f'{uuid}.png' if kind is not BlockType.IMAGE else None,
    )


def _node() -> ContentCorrectionNode:
    counter = MaterialTokenEstimator(_Counter())
    return ContentCorrectionNode(
        object(),
        budget=TokenBudget(counter=counter, token_limit=1000),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        batch_size=8,
    )


def test_correction_collector_removes_by_uuid_without_shifting_later_results():
    first = _block('first', 'corrected first')
    rejected = _block('rejected', 'rejected text')
    later = _block('later', 'corrected later')
    visual = SourceBlock(
        uuid='visual',
        block_type=BlockType.IMAGE,
        content=None,
        image_path='visual.png',
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        ocr_blocks=[
            _block('first', 'original first'),
            rejected,
            _block('later', 'original later'),
            visual,
        ],
        correction_results=[
            ContentCorrectionResult(
                source_block=first, keep=True, cleanup_instructions=[]
            ),
            ContentCorrectionResult(
                source_block=later, keep=True, cleanup_instructions=[]
            ),
            ContentCorrectionResult(
                source_block=rejected, keep=False, cleanup_instructions=[]
            ),
        ],
    )

    blocks = _node().collect(state)['corrected_blocks']

    assert [(block.uuid, block.content) for block in blocks] == [
        ('first', 'corrected first'),
        ('later', 'corrected later'),
        ('visual', None),
    ]
    assert blocks[-1] is visual


def test_neighboring_crop_geometry_respects_selected_physical_page_groups(
    tmp_path,
):
    page_four = tmp_path / 'page-four.png'
    page_nine = tmp_path / 'page-nine.png'
    Image.new('RGB', (10, 10), 'white').save(page_four)
    Image.new('RGB', (10, 20), 'white').save(page_nine)
    first_page_block = SourceBlock(
        uuid='page-four-block',
        block_type=BlockType.PARAGRAPH,
        content='prior physical page',
        crop_path='prior-crop.png',
        crop_bbox=(0, 1, 4, 5),
    )
    target = SourceBlock(
        uuid='target',
        block_type=BlockType.PARAGRAPH,
        content='target on selected page',
        crop_path='target-crop.png',
        crop_bbox=(1, 2, 5, 6),
    )
    following = SourceBlock(
        uuid='following',
        block_type=BlockType.PARAGRAPH,
        content='same page neighbor',
        crop_path='following-crop.png',
        crop_bbox=(2, 4, 6, 8),
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        selected_page_indexes=[4, 9],
        ocr_page_images={4: str(page_four), 9: str(page_nine)},
        ocr_block_pages={'page-four-block': 4, 'target': 9, 'following': 9},
        ocr_blocks=[first_page_block, target, following],
    )

    requests = _node()._requests(state)

    assert [request.source_block.uuid for request in requests] == [
        'page-four-block',
        'target',
        'following',
    ]
    target_request = next(
        request for request in requests if request.source_block.uuid == 'target'
    )
    assert target_request.target_bounds == (0, 0, 4, 4)
