from __future__ import annotations

from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import Source
from kms.source.processing.ocr import OCRNode
from kms.source.processing.ocr_models import OCRArtifact, OCRExtraction
from kms.source.processing.state import SourceProcessingState

SOURCE_UUID = '12345678-1234-5678-1234-567812345678'


class _Provider:
    def __init__(self, extraction: OCRExtraction) -> None:
        self.extraction = extraction
        self.calls = []

    def extract(self, pdf_path, *, source_uuid, pages=None):
        self.calls.append((pdf_path, source_uuid, pages))
        return self.extraction


def test_ocr_node_returns_ordered_blocks_and_independent_selected_page_metadata():
    extraction = OCRExtraction(
        blocks=[
            OCRArtifact(
                page_index=7,
                block_index=1,
                block_type='paragraph',
                content='first in source order',
            ),
            OCRArtifact(
                page_index=4,
                block_index=0,
                block_type='paragraph',
                content='second in source order',
            ),
        ],
        page_indexes=[7, 4, 9],
        page_images={7: 'p7.png', 4: 'p4.png', 9: 'empty-p9.png'},
    )
    provider = _Provider(extraction)
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='book.pdf', pages=[7, 4, 9]),
        source=Source(uuid=SOURCE_UUID, key='book.pdf'),
    )

    result = OCRNode(provider).run(state)

    assert provider.calls == [('book.pdf', SOURCE_UUID, [7, 4, 9])]
    assert [block.content for block in result['ocr_blocks']] == [
        'first in source order',
        'second in source order',
    ]
    assert result['ocr_block_pages'] == {
        'eb737f9b-9172-5e30-b5b7-498739953b88': 7,
        '1b3f5877-c207-52c6-bfd6-1427f893f787': 4,
    }
    assert [block.uuid for block in result['ocr_blocks']] == [
        'eb737f9b-9172-5e30-b5b7-498739953b88',
        '1b3f5877-c207-52c6-bfd6-1427f893f787',
    ]
    assert result['selected_page_indexes'] == [7, 4, 9]
    assert result['ocr_page_images'] == {
        7: 'p7.png',
        4: 'p4.png',
        9: 'empty-p9.png',
    }
