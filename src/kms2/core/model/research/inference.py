"""Typed researcher inputs, actions, observations, and inspected evidence."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.research.retrieval import ResearchEvidenceEntry
from kms2.core.model.source_processing.admission import ResearchSourceMaterial

Position = Annotated[int, Field(ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class WebSearchAction(_StrictModel):
    """Search the web for discovery candidates."""

    kind: Literal['web_search']
    query: str


class FetchWebAction(_StrictModel):
    """Extract selected result URLs and discover passage/image positions."""

    kind: Literal['fetch_web']
    urls: list[str]


class ListEvidenceAction(_StrictModel):
    """List one page of descriptors for a retrieved document."""

    kind: Literal['list_evidence']
    document_position: Position
    start_position: Position


class ReadEvidenceAction(_StrictModel):
    """Read complete selected passages or discovered original images."""

    kind: Literal['read_evidence']
    evidence_positions: list[Position]


class OpenPdfAction(_StrictModel):
    """Open a public PDF without OCRing its pages."""

    kind: Literal['open_pdf']
    url: str


class SearchPdfAction(_StrictModel):
    """Locate pages by native PDF text without citing the snippets."""

    kind: Literal['search_pdf']
    document_position: Position
    query: str


class ReadPdfPagesAction(_StrictModel):
    """OCR and render selected zero-based PDF page indexes."""

    kind: Literal['read_pdf_pages']
    document_position: Position
    page_indexes: list[Position]


class FinishResearchAction(_StrictModel):
    """Return concise sourced findings and inspected evidence positions."""

    kind: Literal['finish']
    brief: str
    evidence_positions: list[Position]


ResearchAction = Annotated[
    WebSearchAction
    | FetchWebAction
    | ListEvidenceAction
    | ReadEvidenceAction
    | OpenPdfAction
    | SearchPdfAction
    | ReadPdfPagesAction
    | FinishResearchAction,
    Field(discriminator='kind'),
]


class ResearchLocalFinding(_StrictModel):
    """Local source or proposition hit with discovery-safe visibility."""

    kind: Literal['block', 'proposition']
    text: str
    inspected: bool


class ResearchTask(_StrictModel):
    """One teacher-written research request without session history."""

    question: str
    context: str
    urls: list[str] = Field(default_factory=list)
    local_findings: list[ResearchLocalFinding] = Field(default_factory=list)


class ResearchFailure(_StrictModel):
    """An expected resource failure that cannot support a factual claim."""

    operation: str
    url: str | None = None
    message: str


class ResearchDocumentDescriptor(_StrictModel):
    """Model-facing document metadata without private bytes or paths."""

    position: Position
    url: str
    title: str | None
    media_type: str
    page_count: int | None


class ResearchEvidenceDescriptor(_StrictModel):
    """Paginated discovery descriptor; previews are not inspected evidence."""

    position: Position
    document_position: Position
    kind: Literal['web_passage', 'image', 'pdf_page']
    title: str | None
    url: str
    page_index: int | None
    line_start: int | None
    line_end: int | None
    token_count: int
    heading_path: list[str] = Field(default_factory=list)
    preview: str = ''


class InspectedResearchEvidence(_StrictModel):
    """Evidence actually included in a completed researcher prediction."""

    position: Position
    kind: Literal['web_passage', 'image', 'pdf_page']
    url: str
    title: str | None
    heading_path: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    page_index: int | None
    line_start: int | None
    line_end: int | None
    raw_content: str | None
    content: str | None
    native_image_position: Position | None


class ResearchToolObservation(_StrictModel):
    """One completed research action and its non-authoritative observations."""

    kind: Literal[
        'web_search',
        'fetch_web',
        'list_evidence',
        'read_evidence',
        'open_pdf',
        'search_pdf',
        'read_pdf_pages',
    ]
    summary: str
    document_positions: list[Position] = Field(default_factory=list)
    evidence_positions: list[Position] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)


class ResearchContext(_StrictModel):
    """Serialized researcher state; private document payloads stay in nodes."""

    task: ResearchTask
    documents: list[ResearchDocumentDescriptor] = Field(default_factory=list)
    evidence: list[ResearchEvidenceDescriptor] = Field(default_factory=list)
    observations: list[ResearchToolObservation] = Field(default_factory=list)
    tools_remaining: int
    inspected_evidence: list[InspectedResearchEvidence] = Field(
        default_factory=list
    )


class ResearchResult(_StrictModel):
    """Final sourced brief and evidence selected for the teacher."""

    brief: str
    selected_evidence: list[ResearchEvidenceEntry] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)
    source_materials: list[ResearchSourceMaterial] = Field(default_factory=list)
