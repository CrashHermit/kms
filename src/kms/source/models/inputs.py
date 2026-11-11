"""Typed inputs for source processing and acquisition."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PDFSourceInput(_StrictModel):
    """An original PDF handled by the configured OCR provider."""

    kind: Literal['pdf'] = 'pdf'
    pdf_path: str
    pages: list[int] | None = None


class CapturedPDFSource(_StrictModel):
    """Full original PDF capture before source processing or inspection."""

    url: str | None = None
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    data: bytes
