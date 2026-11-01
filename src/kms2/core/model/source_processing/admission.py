"""Contracts for source snapshot preparation and researched materials."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.input import SourceProcessingInput
from kms2.core.model.source_processing.ocr import OCRPageArtifact


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class OwnedSourceSnapshot(_StrictModel):
    """Persisted owned source and its ordered structural pages."""

    kind: Literal['owned'] = 'owned'
    source: Source
    pages: list[SourcePage]


class StagedSourceSnapshot(_StrictModel):
    """Processed source snapshot awaiting atomic publication with a turn."""

    kind: Literal['staged'] = 'staged'
    source: Source
    pages: list[SourcePage]


SourceAdmissionResult = Annotated[
    OwnedSourceSnapshot | StagedSourceSnapshot,
    Field(discriminator='kind'),
]


class SourceSnapshotPreparation(_StrictModel):
    """Captured source identity and typed input awaiting processing."""

    source: Source
    input: SourceProcessingInput


class ResearchImageMaterial(_StrictModel):
    """Inspected downloaded image capture and normalized PNG asset."""

    url: str
    resolved_url: str | None = None
    retrieved_at: datetime
    media_bytes: bytes
    original_media_bytes: bytes | None = None
    original_media_type: str | None = None


class ResearchWebMaterial(_StrictModel):
    """Full captured web page and only its inspected downloaded images."""

    kind: Literal['web'] = 'web'
    url: str
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    raw_markdown: str
    images: list[ResearchImageMaterial] = Field(default_factory=list)


class ResearchPDFPageMaterial(_StrictModel):
    """One selected, inspected original PDF page and its OCR artifacts."""

    page_index: int = Field(ge=0)
    markdown: str
    artifacts: OCRPageArtifact
    rendered_page: bytes


class ResearchPDFMaterial(_StrictModel):
    """Selected inspected pages of a captured PDF document."""

    kind: Literal['pdf'] = 'pdf'
    url: str
    resolved_url: str | None = None
    title: str | None = None
    retrieved_at: datetime
    pages: list[ResearchPDFPageMaterial]


ResearchSourceMaterial = Annotated[
    ResearchWebMaterial | ResearchPDFMaterial,
    Field(discriminator='kind'),
]


class ResearchSourceMigrationResult(_StrictModel):
    """Counts historical research records migrated into source snapshots."""

    migrated_evidence_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
