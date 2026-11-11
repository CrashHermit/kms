"""LangGraph OCR node for KMS2."""

from dataclasses import dataclass

from kms.ocr.provider import OCRProvider, source_pages_from_ocr
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import SourcePage
from kms.source.processing.state import SourceProcessingState


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage] | dict[int, str]]:
        """Extract ordered source pages and blocks for the PDF."""
        source_input = state.input
        assert isinstance(source_input, PDFSourceInput)
        artifacts = self.provider.extract(
            source_input.pdf_path,
            source_uuid=state.source.uuid,
            pages=source_input.pages,
        )
        return {
            'ocr_pages': source_pages_from_ocr(
                artifacts, source_uuid=state.source.uuid
            ),
            'ocr_page_images': {
                page.page_index: page.page_image_path for page in artifacts
            },
        }
