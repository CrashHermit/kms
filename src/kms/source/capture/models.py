"""Typed URL inspection and link-only PDF results."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class URLInspection(_StrictModel):
    """Metadata for a safely inspected external URL."""

    requested_url: str
    final_url: str
    retrieved_at: datetime
    media_type: str


class PDFCandidate(_StrictModel):
    """A PDF URL available for explicit manual handling."""

    url: str
    title: str | None = None
