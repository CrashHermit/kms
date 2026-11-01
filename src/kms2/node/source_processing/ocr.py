"""LangGraph OCR node for KMS2."""

from dataclasses import dataclass

from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.input import PDFSourceInput
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.ocr.provider import OCRProvider, source_pages_from_ocr


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(self, state: SourceProcessingState) -> dict[str, list[SourcePage]]:
        """Extract ordered source pages and blocks for the PDF."""
        source_input = state.input
        assert isinstance(source_input, PDFSourceInput)
        pages = self.provider.extract(
            source_input.pdf_path,
            source_uuid=state.source.uuid,
            pages=source_input.pages,
        )
        return {'ocr_pages': source_pages_from_ocr(pages)}
