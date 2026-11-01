"""Private invocation state and stable position registries for research."""

from pydantic import BaseModel, Field

from kms2.core.model.research.inference import (
    ResearchAction,
    ResearchFailure,
    ResearchResult,
    ResearchTask,
    ResearchToolObservation,
)
from kms2.core.model.research.retrieval import (
    ResearchDocument,
    ResearchEvidenceEntry,
)
from kms2.core.model.source_processing.admission import ResearchSourceMaterial


class ResearchState(BaseModel):
    """Own documents, evidence, and stable positions for one research task."""

    task: ResearchTask
    max_tool_calls: int = Field(ge=0)
    catalog_page_size: int = Field(gt=0)
    documents: list[ResearchDocument] = Field(default_factory=list)
    visible_evidence_positions: list[int] = Field(default_factory=list)
    search_titles: dict[str, str | None] = Field(default_factory=dict)
    document_positions: dict[str, int] = Field(default_factory=dict)
    evidence: list[ResearchEvidenceEntry] = Field(default_factory=list)
    evidence_positions: dict[str, int] = Field(default_factory=dict)
    inspected_positions: list[int] = Field(default_factory=list)
    pending_positions: list[int] = Field(default_factory=list)
    observations: list[ResearchToolObservation] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)
    executed_tool_calls: int = 0
    latest_action: ResearchAction | None = None
    final_result: ResearchResult | None = None
    source_materials: list[ResearchSourceMaterial] = Field(default_factory=list)

    def register_document(self, document: ResearchDocument) -> int:
        """Register a document once, preserving its first stable position."""
        key = document.url
        if key in self.document_positions:
            return self.document_positions[key]
        position = len(self.documents)
        self.documents.append(document)
        self.document_positions[key] = position
        return position

    def register_evidence(self, entry: ResearchEvidenceEntry) -> int:
        """Register evidence by document and source coordinates exactly once."""
        key = _evidence_key(entry)
        if key in self.evidence_positions:
            return self.evidence_positions[key]
        position = len(self.evidence)
        self.evidence.append(entry.model_copy(update={'position': position}))
        self.evidence_positions[key] = position
        return position

    def entry(self, position: int) -> ResearchEvidenceEntry:
        """Return the evidence at one stable invocation position."""
        return self.evidence[position]


def _evidence_key(entry: ResearchEvidenceEntry) -> str:
    """Build a stable content-independent key from source coordinates."""
    if entry.kind == 'image':
        coordinate = entry.resolved_url or entry.url
    elif entry.kind == 'pdf_page':
        coordinate = f'{entry.page_index}:{entry.line_start}:{entry.line_end}'
    else:
        coordinate = f'{entry.line_start}:{entry.line_end}'
    return f'{entry.document_position}:{entry.kind}:{coordinate}'
