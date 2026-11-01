"""Transient provider and document-reader models for external research."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.source_processing.ocr import OCRPageArtifact


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class WebSearchHit(_StrictModel):
    """One provider search result; its snippet is discovery-only."""

    title: str | None
    url: str
    snippet: str
    score: float


class WebExtraction(_StrictModel):
    """Provider Markdown and image links received for one URL."""

    url: str
    retrieved_at: datetime
    raw_markdown: str
    image_urls: list[str] = Field(default_factory=list)


class WebExtractionFailure(_StrictModel):
    """One URL that Tavily could not extract."""

    url: str
    error: str


class WebExtractionBatch(_StrictModel):
    """Independent Tavily successes and URL-specific failures."""

    results: list[WebExtraction] = Field(default_factory=list)
    failures: list[WebExtractionFailure] = Field(default_factory=list)


class DownloadedMedia(_StrictModel):
    """One bounded public image or PDF download."""

    requested_url: str
    final_url: str
    retrieved_at: datetime
    media_type: str
    data: bytes


class ResearchDocument(_StrictModel):
    """Private invocation document data; never expose paths to inference."""

    url: str
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    media_type: str
    raw_markdown: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    pdf_path: Path | None = None
    page_count: int | None = Field(default=None, ge=0)


class ResearchEvidenceEntry(_StrictModel):
    """Private evidence record with invocation-local registry coordinates."""

    document_position: int = Field(ge=0)
    position: int | None = Field(default=None, ge=0)
    kind: Literal['web_passage', 'image', 'pdf_page']
    url: str
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    page_index: int | None = Field(default=None, ge=0)
    line_start: int | None = Field(default=None, ge=0)
    line_end: int | None = Field(default=None, ge=0)
    heading_path: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    preview: str = ''
    raw_content: str | None = None
    content: str | None = None
    media_type: str | None = None
    media_bytes: bytes | None = None
    original_media_type: str | None = None
    original_media_bytes: bytes | None = None
    ocr_artifacts: OCRPageArtifact | None = None
    rendered_page_bytes: bytes | None = None
    inspected: bool = False


class PdfSearchMatch(_StrictModel):
    """Native-text PDF locator result, not inspected cited evidence."""

    page_index: int = Field(ge=0)
    snippet: str


class PdfSearchResult(_StrictModel):
    """Page matches and whether the configured result limit truncated them."""

    matches: list[PdfSearchMatch] = Field(default_factory=list)
    truncated: bool = False
