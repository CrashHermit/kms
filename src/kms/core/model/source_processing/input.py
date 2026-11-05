"""Typed inputs accepted by the shared source-processing pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PDFSourceInput(_StrictModel):
    """An original PDF handled by the configured OCR provider."""

    kind: Literal['pdf'] = 'pdf'
    pdf_path: str
    pages: list[int] | None = None
