"""Transient provider models for external research retrieval."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class WebSearchHit(_StrictModel):
    """One provider search result; its snippet is discovery-only."""

    title: str | None
    url: str
    snippet: str


class DownloadedMedia(_StrictModel):
    """Classified public URL with PDF bytes when the resource is a PDF."""

    requested_url: str
    final_url: str
    retrieved_at: datetime
    media_type: str
    data: bytes | None
