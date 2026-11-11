"""Deterministic source-processing node boundary coverage."""

import asyncio

from PIL import Image

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.ocr.provider import OCRProvider
from kms.source.models.source import Source, SourceBlock, SourcePage
from kms.source.processing.correction.models import ContentCorrectionResult
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.ocr import OCRNode
from kms.source.processing.ocr_models import OCRArtifact, OCRPageArtifact
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


def _content_correction_node(corrector: object) -> ContentCorrectionNode:
    return ContentCorrectionNode(
        corrector,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=10_000
        ),
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=32,
    )


def test_ocr_node_delegates_once_to_injected_provider(tmp_path):
    calls = []
    page_image = tmp_path / 'page.png'
    Image.new('RGB', (4, 4), 'white').save(page_image)
    artifacts = [
        OCRPageArtifact(
            page_index=2,
            page_image_path=str(page_image),
            markdown='# Page 2',
            blocks=[
                OCRArtifact(
                    page_index=2,
                    block_index=1,
                    block_type='paragraph',
                    content='second',
                    crop_bbox=(1, 1, 2, 2),
                ),
                OCRArtifact(
                    page_index=2,
                    block_index=0,
                    block_type='paragraph',
                    content='first',
                    crop_bbox=(0, 0, 1, 1),
                ),
            ],
        )
    ]

    class FakeProvider:
        def extract(
            self,
            pdf_path: str,
            *,
            source_uuid: str,
            pages: list[int] | None = None,
        ) -> list[OCRPageArtifact]:
            calls.append((pdf_path, source_uuid, pages))
            return artifacts

    provider: OCRProvider = FakeProvider()
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf', 'pages': [2]},
        source=Source(
            uuid='12345678-1234-5678-1234-567812345678', key='book.pdf'
        ),
    )
    result = OCRNode(provider).run(state)

    assert calls == [('book.pdf', '12345678-1234-5678-1234-567812345678', [2])]
    assert set(result) == {'ocr_pages', 'ocr_page_images'}
    assert result['ocr_pages'][0].index == 2
    assert result['ocr_page_images'] == {2: str(page_image)}
    assert [block.content for block in result['ocr_pages'][0].blocks] == [
        'first',
        'second',
    ]
    assert 'ocr_artifacts' not in result


def test_content_correction_dispatch_routes_empty_ocr_to_collect():
    node = _content_correction_node(object())
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
    )

    assert asyncio.run(node.dispatch(state)) == 'content_correction_collect'


def test_content_correction_collects_results_by_page_and_block_position():
    first = SourceBlock(uuid='first', block_type='paragraph', content='first')
    second = SourceBlock(
        uuid='second', block_type='paragraph', content='second'
    )
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        ocr_pages=[SourcePage(index=4, blocks=[first, second])],
        source=Source(uuid='source-1', key='book.pdf'),
    )
    state.correction_results = [
        ContentCorrectionResult(
            page_index=4,
            block_position=1,
            source_block=second.model_copy(
                update={'content': 'corrected second'}
            ),
            keep=True,
            cleanup_instructions=[],
        ),
        ContentCorrectionResult(
            page_index=4,
            block_position=0,
            source_block=first.model_copy(
                update={'content': 'corrected first'}
            ),
            keep=True,
            cleanup_instructions=[],
        ),
    ]

    result = _content_correction_node(object()).collect(state)

    assert [block.content for block in result['corrected_pages'][0].blocks] == [
        'corrected first',
        'corrected second',
    ]
