"""LangGraph OCR node for KMS2."""

from dataclasses import dataclass

from kms2.core.model import SourceBlock, SourcePage, VisualAsset
from kms2.langgraph.source.state import SourceState
from kms2.ocr.provider import OCRProvider


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Extract ordered source pages and blocks for the PDF."""
        pages = self.provider.extract(
            state.pdf_path,
            pages=state.pages,
        )
        ocr_pages = [
            SourcePage(
                index=page.page_index,
                markdown=page.markdown,
                blocks=[
                    SourceBlock(
                        block_type=artifact.block_type,
                        content=artifact.content,
                        crop_path=artifact.crop_path,
                        crop_bbox=artifact.crop_bbox,
                        assets=[
                            VisualAsset(path=image.path)
                            for image in artifact.images
                        ],
                    )
                    for artifact in sorted(
                        page.blocks, key=lambda item: item.block_index
                    )
                ],
            )
            for page in pages
        ]
        return {'ocr_pages': ocr_pages}
