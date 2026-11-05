"""Local-first evidence acquisition requests and outcome projections."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.source import SourceRevisionReference
from kms.core.model.teaching.positions import Position


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class LocalEvidenceRequest(_StrictModel):
    """Search owned source blocks and canonical propositions for one need."""

    kind: Literal['local'] = 'local'
    question: str
    context: str
    require_visual_inspection: bool = False


class ExternalEvidenceRequest(_StrictModel):
    """Expand one previously completed local outcome through research."""

    kind: Literal['external'] = 'external'
    local_outcome_position: Position


EvidenceAcquisitionRequest = Annotated[
    LocalEvidenceRequest | ExternalEvidenceRequest,
    Field(discriminator='kind'),
]


class AcquireEvidenceAction(_StrictModel):
    """Search local evidence or explicitly expand a local outcome externally."""

    kind: Literal['acquire_evidence'] = 'acquire_evidence'
    request: EvidenceAcquisitionRequest


class EvidenceResearchOutcome(_StrictModel):
    """Failures and persisted source snapshots from one external expansion."""

    failures: list[str]
    source_revisions: list[SourceRevisionReference] = Field(
        default_factory=list
    )


class LocalEvidenceOutcome(_StrictModel):
    """Private local-search result and optional external expansion state."""

    request: LocalEvidenceRequest
    block_positions: list[Position]
    proposition_positions: list[Position]
    expanded: bool = False
    research: EvidenceResearchOutcome | None = None


class TeachingLocalOutcome(_StrictModel):
    """Local outcome and persisted research identities for one teacher
    prediction."""

    position: Position
    question: str
    block_positions: list[Position]
    proposition_positions: list[Position]
    expanded: bool
    research: EvidenceResearchOutcome | None
    require_visual_inspection: bool = False
