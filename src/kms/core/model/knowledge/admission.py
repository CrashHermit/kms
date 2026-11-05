"""Request-local extraction drafts and prepared knowledge writes."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import (
    KnowledgeAssertion,
    Proposition,
)
from kms.core.model.source import SourceRevisionReference


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class KnowledgeEvidenceBlock(_StrictModel):
    """Inspected source block offered as possible support for extraction."""

    position: int
    source_key: str
    page_indexes: list[int]
    source_block: SourceBlockContext


class KnowledgeExistingProposition(_StrictModel):
    """A proposition explicitly read in the current teaching invocation."""

    position: int
    material: MultimodalContent


class KnowledgeResponseSegment(_StrictModel):
    """Response passage paired only with evidence explicitly cited for it."""

    material: MultimodalContent
    block_positions: list[int]
    image_positions: list[int]
    proposition_positions: list[int]


class KnowledgeExtractionInput(_StrictModel):
    """Private-state-free request for grounded proposition extraction."""

    segments: list[KnowledgeResponseSegment]
    blocks: list[KnowledgeEvidenceBlock]
    existing_propositions: list[KnowledgeExistingProposition]


class ExistingPropositionReference(_StrictModel):
    """Reference to an inspected proposition in the extraction request."""

    kind: Literal['existing']
    position: int


class DraftPropositionReference(_StrictModel):
    """Reference to a proposition produced in this extraction result."""

    kind: Literal['draft']
    position: int


PropositionReference = Annotated[
    ExistingPropositionReference | DraftPropositionReference,
    Field(discriminator='kind'),
]


class PropositionDraft(_StrictModel):
    """One self-contained proposition and direct inspected block support."""

    content: str
    supporting_block_positions: list[int]


class ArgumentDraft(_StrictModel):
    """One joint-premise argument with one conclusion."""

    explanation: str
    premises: list[PropositionReference]
    conclusion: PropositionReference
    supporting_block_positions: list[int]


class KnowledgeDraft(_StrictModel):
    """Model-produced claims and arguments without persistence identities."""

    propositions: list[PropositionDraft]
    arguments: list[ArgumentDraft]


class PropositionEquivalenceDecision(_StrictModel):
    """Whether one candidate expresses exactly the target proposition."""

    position: int
    is_equivalent: bool


class PropositionIdentityCandidate(_StrictModel):
    """Positioned bounded candidate supplied to equivalence judgment."""

    position: int
    content: str


class PropositionEquivalenceInput(_StrictModel):
    """Target and bounded candidates for semantic identity judgment."""

    proposition_content: str
    candidates: list[PropositionIdentityCandidate]


class KnowledgeAssertionWrite(_StrictModel):
    """Assertion transport before the containing teacher turn is committed."""

    assertion: KnowledgeAssertion
    proposition_uuid: str
    supporting_block_uuids: list[str]
    argument_uuids: list[str]


class KnowledgeAdmission(_StrictModel):
    """Prepared immutable graph writes, rechecked at teaching commit."""

    expected_revision: int
    source_revisions: list[SourceRevisionReference]
    new_propositions: list[Proposition]
    proposition_uuids: list[str]
    assertions: list[KnowledgeAssertionWrite]
    new_arguments: list[ArgumentRecord]
    existing_argument_uuids: list[str]


class ResolvedPropositionMatchExisting(_StrictModel):
    """A draft proposition resolved to an owned canonical node."""

    kind: Literal['existing']
    uuid: str


class ResolvedPropositionMatchDraft(_StrictModel):
    """A draft proposition resolved to an earlier draft representative."""

    kind: Literal['draft']
    position: int


ResolvedPropositionMatch = Annotated[
    ResolvedPropositionMatchExisting | ResolvedPropositionMatchDraft,
    Field(discriminator='kind'),
]


class ResolvedPropositionDraft(_StrictModel):
    """A model claim with vector and identity result from preparation."""

    draft: PropositionDraft
    embedding: list[float]
    match: ResolvedPropositionMatch | None


class ResolvedArgumentDraft(_StrictModel):
    """Argument draft with embeddings of its canonical complete claims."""

    draft: ArgumentDraft
    embedding: list[float]


class ResolvedKnowledgeDraft(_StrictModel):
    """Identity-resolved extraction state for backend record preparation."""

    propositions: list[ResolvedPropositionDraft]
    arguments: list[ResolvedArgumentDraft]
    existing_proposition_uuids: list[str]
    supporting_block_uuids: list[str]
    block_source_revisions: list[SourceRevisionReference]


__all__ = [
    'DraftPropositionReference',
    'ExistingPropositionReference',
    'KnowledgeAdmission',
    'KnowledgeAssertionWrite',
    'KnowledgeDraft',
    'KnowledgeEvidenceBlock',
    'KnowledgeExistingProposition',
    'KnowledgeExtractionInput',
    'KnowledgeResponseSegment',
    'PropositionDraft',
    'PropositionEquivalenceDecision',
    'PropositionEquivalenceInput',
    'PropositionIdentityCandidate',
    'PropositionReference',
    'ArgumentDraft',
    'ResolvedKnowledgeDraft',
    'ResolvedPropositionDraft',
    'ResolvedArgumentDraft',
    'ResolvedPropositionMatch',
]
