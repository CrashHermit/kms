"""Typed inputs accepted by the shared source-processing pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.page import SourcePage


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PDFSourceInput(_StrictModel):
    """An original PDF handled by the configured OCR provider."""

    kind: Literal['pdf'] = 'pdf'
    pdf_path: str
    pages: list[int] | None = None


class OCRSourceInput(_StrictModel):
    """Already OCRed pages retaining real block and crop artifacts."""

    kind: Literal['ocr'] = 'ocr'
    pages: list[SourcePage]


class TextSourceInput(_StrictModel):
    """Prepared text pages that bypass visual content correction."""

    kind: Literal['text'] = 'text'
    pages: list[SourcePage]


SourceProcessingInput = Annotated[
    PDFSourceInput | OCRSourceInput | TextSourceInput,
    Field(discriminator='kind'),
]
