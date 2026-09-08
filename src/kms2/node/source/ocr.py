"""LangGraph OCR node for KMS2."""

from collections import defaultdict
from dataclasses import dataclass

from kms2.core.model import SourceBlock, SourcePage, VisualAsset
from kms2.core.model.ocr import OCRArtifact
from kms2.langgraph.source.state import SourceState
from kms2.ocr.provider import OCRProvider


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Extract ordered source pages and blocks for the PDF."""
        artifacts = self.provider.extract(
            state.pdf_path,
            pages=state.pages,
        )
        artifacts_by_page: dict[int, list[OCRArtifact]] = defaultdict(list)
        for artifact in artifacts:
            artifacts_by_page[artifact.page_index].append(artifact)

        ocr_pages: list[SourcePage] = []
        for page_index, page_artifacts in sorted(artifacts_by_page.items()):
            blocks = [
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
                    page_artifacts, key=lambda item: item.block_index
                )
            ]
            ocr_pages.append(SourcePage(index=page_index, blocks=blocks))
        return {'ocr_pages': ocr_pages}
