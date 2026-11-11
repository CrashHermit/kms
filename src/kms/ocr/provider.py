"""Provider-neutral OCR interfaces for KMS2."""

from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.ocr_models import OCRPageArtifact


def source_pages_from_ocr(
    artifacts: list[OCRPageArtifact], *, source_uuid: str
) -> list[SourcePage]:
    """Map provider artifacts to canonical source pages."""
    namespace = UUID(source_uuid)
    pages: list[SourcePage] = []
    for page_ordinal, page in enumerate(artifacts):
        blocks: list[SourceBlock] = []
        for block_ordinal, artifact in enumerate(
            sorted(page.blocks, key=lambda item: item.block_index)
        ):
            is_image = artifact.block_type is BlockType.IMAGE
            blocks.append(
                SourceBlock(
                    uuid=str(
                        uuid5(
                            namespace,
                            f'ocr:block:{page_ordinal}:{block_ordinal}',
                        )
                    ),
                    block_type=artifact.block_type,
                    content=artifact.content,
                    crop_path=artifact.crop_path,
                    crop_bbox=artifact.crop_bbox,
                    image_path=artifact.crop_path if is_image else None,
                    image_page_indexes=[artifact.page_index]
                    if is_image
                    else [],
                )
            )
        pages.append(
            SourcePage(
                index=page.page_index,
                markdown=page.markdown,
                blocks=blocks,
            )
        )
    return pages


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
