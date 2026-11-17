"""Provider-neutral OCR interfaces for KMS2."""

from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from kms.source.models.source import BlockType, SourceBlock
from kms.source.processing.ocr_models import OCRExtraction


def source_blocks_from_ocr(
    extraction: OCRExtraction, *, source_uuid: str
) -> list[SourceBlock]:
    """Map flat provider artifacts to canonical source blocks."""
    namespace = UUID(source_uuid)
    return [
        SourceBlock(
            uuid=str(uuid5(namespace, f'ocr:block:{flat_position}')),
            block_type=artifact.block_type,
            content=artifact.content,
            crop_path=artifact.crop_path,
            crop_bbox=artifact.crop_bbox,
            image_path=artifact.crop_path
            if artifact.block_type is BlockType.IMAGE
            else None,
        )
        for flat_position, artifact in enumerate(extraction.blocks)
    ]


class OCRProvider(Protocol):
    """Extract flat provider-neutral OCR artifacts from a source PDF."""

    def extract(
        self,
        pdf_path: str | Path,
        *,
        source_uuid: str,
        pages: list[int] | None = None,
    ) -> OCRExtraction:
        """Extract blocks and physical-page metadata for a PDF."""
        ...
