"""Provider-neutral OCR interfaces for KMS2."""

from pathlib import Path
from typing import Protocol

from kms2.core.model import OCRArtifact


class OCRProvider(Protocol):
    """Extracts provider-neutral OCR artifacts from a source PDF."""

    def extract(
        self,
        pdf_path: str | Path,
        *,
        pages: list[int] | None = None,
    ) -> list[OCRArtifact]:
        """Extract ordered OCR artifacts for a PDF and optional page subset."""
        ...
