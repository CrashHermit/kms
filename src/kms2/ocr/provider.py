"""Provider-neutral OCR interfaces for KMS2."""

from pathlib import Path
from typing import Protocol

from kms2.core.model.source_stage.ocr import OCRPageArtifact


class OCRProvider(Protocol):
    """Extracts provider-neutral OCR pages from a source PDF."""

    def extract(
        self,
        pdf_path: str | Path,
        *,
        pages: list[int] | None = None,
    ) -> list[OCRPageArtifact]:
        """Extract ordered OCR pages for a PDF and optional page subset."""
        ...
