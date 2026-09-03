from kms2.core.model import OCRArtifact, Source, SourceContent
from kms2.langgraph.source import OCRNode, SourceState
from kms2.ocr.provider import OCRProvider


def test_source_state_carries_source_pipeline_data():
    source = Source(uuid='source-1', key='document.pdf')
    artifact = OCRArtifact(
        page_index=0,
        block_index=0,
        block_type='text',
        content='original',
        crop_path='output/blocks/block-0.png',
    )
    content_corrector = [
        SourceContent(
            uuid='content-1',
            content='corrected source content',
        )
    ]
    state: SourceState = {
        'pdf_path': 'document.pdf',
        'pages': [1, 2],
        'source': source,
        'ocr_artifacts': [artifact],
        'content_corrector': content_corrector,
    }

    assert state['source'] is source
    assert state['pages'] == [1, 2]
    assert state['ocr_artifacts'] == [artifact]
    assert state['content_corrector'] == content_corrector


def test_ocr_node_delegates_once_to_injected_provider():
    calls = []
    artifact = OCRArtifact(
        page_index=2,
        block_index=0,
        block_type='text',
        content='original',
    )

    class FakeProvider:
        def extract(
            self, pdf_path: str, *, pages: list[int] | None = None
        ) -> list[OCRArtifact]:
            calls.append((pdf_path, pages))
            return [artifact]

    provider: OCRProvider = FakeProvider()
    result = OCRNode(provider).run({'pdf_path': 'book.pdf', 'pages': [2]})

    assert calls == [('book.pdf', [2])]
    assert result == {'ocr_artifacts': [artifact]}
    assert 'ocr_response' not in result
    assert 'raw_response' not in result
