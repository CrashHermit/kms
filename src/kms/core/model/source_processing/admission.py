"""Contracts for source snapshot preparation and captured acquisition data."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.input import PDFSourceInput


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceSnapshot(_StrictModel):
    """Complete source content, whether newly processed or already owned."""

    source: Source
    pages: list[SourcePage]


class SourceSnapshotPreparation(_StrictModel):
    """Captured source identity and typed input awaiting processing."""

    source: Source
    input: PDFSourceInput


class CapturedPDFSource(_StrictModel):
    """Full original PDF capture before source processing or inspection."""

    kind: Literal['pdf'] = 'pdf'
    url: str
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    data: bytes
