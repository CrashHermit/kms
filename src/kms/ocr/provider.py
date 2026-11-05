"""Provider-neutral OCR interfaces for KMS2."""

from pathlib import Path
from typing import Protocol

from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source_processing.ocr import OCRPageArtifact
from kms.core.model.visual_asset import VisualAsset


def source_pages_from_ocr(
    artifacts: list[OCRPageArtifact],
) -> list[SourcePage]:
    """Map provider artifacts to canonical source pages."""
    return [
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
        for page in artifacts
    ]


class OCRProvider(Protocol):
    """Extracts provider-neutral OCR pages from a source PDF."""

    def extract(
        self,
        pdf_path: str | Path,
        *,
        source_uuid: str,
        pages: list[int] | None = None,
    ) -> list[OCRPageArtifact]:
        """Extract ordered OCR pages for a PDF and optional page subset."""
        ...
