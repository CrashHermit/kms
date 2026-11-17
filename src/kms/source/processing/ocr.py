"""LangGraph OCR node for KMS2."""

from dataclasses import dataclass

from kms.ocr.provider import OCRProvider, source_blocks_from_ocr
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import SourceBlock
from kms.source.processing.state import SourceProcessingState


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(
        self, state: SourceProcessingState
    ) -> dict[
        str,
        list[SourceBlock] | dict[int, str] | dict[str, int] | list[int],
    ]:
        """Extract blocks and physical-page capture metadata."""
        source_input = state.input
        assert isinstance(source_input, PDFSourceInput)
        extraction = self.provider.extract(
            source_input.pdf_path,
            source_uuid=state.source.uuid,
            pages=source_input.pages,
        )
        blocks = source_blocks_from_ocr(
            extraction, source_uuid=state.source.uuid
        )
        return {
            'ocr_blocks': blocks,
            'ocr_block_pages': {
                block.uuid: artifact.page_index
                for block, artifact in zip(
                    blocks, extraction.blocks, strict=True
                )
            },
            'selected_page_indexes': extraction.page_indexes,
            'ocr_page_images': extraction.page_images,
        }
