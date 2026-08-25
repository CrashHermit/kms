"""Core dataclasses shared across the ingestion pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextNodeInput(BaseModel):
    """Text-only source node crossing a DSPy enrichment boundary."""

    local_index: int = Field(description='Position in the composed input.')
    node_type: str = Field(description='Canonical node type.')
    node_text: str = Field(description='Canonical node text.')



class HubMentionInput(BaseModel):
    """Text-only hub mention crossing an adjudication boundary."""

    model_config = ConfigDict(extra='forbid')

    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> 'HubMentionInput':
        name = record['name']
        aliases = [
            alias
            for alias in (record.get('aliases') or [])
            if alias != name
        ]
        return cls(
            name=name,
            aliases=aliases,
            description=record.get('description'),
        )


class SearchQuery(BaseModel):
    """One ordered, text-only query accepted by the search pipeline."""

    model_config = ConfigDict(extra='forbid')

    parts: list[TextNodeInput]

    @model_validator(mode='after')
    def validate_parts(self) -> 'SearchQuery':
        if not self.parts:
            raise ValueError('search query must contain at least one part')
        expected = list(range(len(self.parts)))
        actual = [part.local_index for part in self.parts]
        if actual != expected:
            raise ValueError(
                'search query local_index values must be contiguous from zero'
            )
        if any(not part.node_text.strip() for part in self.parts):
            raise ValueError('search query parts must contain non-blank text')
        return self

@dataclass(frozen=True, slots=True)
class StatementEnrichmentInput:
    """Complete typed input for one statement enrichment operation."""

    statement_uuid: str
    statement: tuple[TextNodeInput, ...]
    canonical_knowledge: str


@dataclass(frozen=True, slots=True)
class ProcedureEnrichmentInput:
    """Typed statement and procedure content for canonicalization or generation."""

    source: str
    statement_uuid: str
    statement: tuple[TextNodeInput, ...]
    procedure_uuid: str | None = None
    procedure: tuple[TextNodeInput, ...] | None = None
    canonical_knowledge: str = ''


@dataclass(frozen=True, slots=True)
class HubRecord:
    """One source-local component or semantic hub candidate."""

    uuid: str
    name: str
    description: str | None
    embedding: list[float]
    source: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HubBuildBundle:
    """Source-scoped inputs for one incremental hub assignment stage."""

    source: str
    components: tuple[HubRecord, ...] = ()
    candidate_hubs: tuple[HubRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class HubComponent:
    """One source-local entity or predicate occurrence."""

    uuid: str
    source: str
    node_id: int
    name: str
    description: str | None
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class TripletMembership:
    """Canonical hub memberships for one ordered triplet occurrence."""

    triplet_index: int
    source: str
    subject_hubs: tuple[str, ...]
    predicate_hubs: tuple[str, ...]
    object_hubs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StatementHubRecord:
    """One enriched statement available for source-local clustering."""

    uuid: str
    source: str
    description: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class ProcedureHubRecord:
    """One enriched procedure available for source-local clustering."""

    uuid: str
    source: str
    description: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class CanonicalConcept:
    """Canonical subject or object concept used by one assertion."""

    uuid: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class CanonicalPredicate:
    """Canonical predicate used by one assertion."""

    uuid: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class KnowledgeAssertion:
    """A complete canonical assertion with source evidence."""

    uuid: str
    name: str
    description: str
    subject: CanonicalConcept
    predicate: CanonicalPredicate
    object: CanonicalConcept
    evidence_node_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class KnowledgeIndex:
    """Source-scoped canonical assertions available during construction."""

    source: str
    assertions: tuple[KnowledgeAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class Knowledge:
    """A deterministic selection of canonical assertions."""

    assertions: tuple[KnowledgeAssertion, ...] = ()

    def union(self, other: 'Knowledge') -> 'Knowledge':
        """Combines assertions by UUID and merges their evidence."""
        by_uuid = {assertion.uuid: assertion for assertion in self.assertions}
        for assertion in other.assertions:
            existing = by_uuid.get(assertion.uuid)
            if existing is None:
                by_uuid[assertion.uuid] = assertion
            elif existing != assertion:
                by_uuid[assertion.uuid] = KnowledgeAssertion(
                    uuid=existing.uuid,
                    name=existing.name,
                    description=existing.description,
                    subject=existing.subject,
                    predicate=existing.predicate,
                    object=existing.object,
                    evidence_node_ids=(
                        existing.evidence_node_ids | assertion.evidence_node_ids
                    ),
                )
        return Knowledge(
            assertions=tuple(
                sorted(by_uuid.values(), key=lambda item: item.uuid)
            )
        )

    def render(self) -> str:
        """Renders complete assertions for an LLM input boundary."""
        return '\n'.join(
            f'{assertion.subject.name} --{assertion.predicate.name}--> '
            f'{assertion.object.name}: {assertion.description}'
            for assertion in self.assertions
        )


class CardStatus(StrEnum):
    """Lifecycle status of a card."""

    ACTIVE = 'active'
    SUSPENDED = 'suspended'
    BURIED = 'buried'
    DONE = 'done'


class Rating(StrEnum):
    """FSRS rating values (Again, Hard, Good, Easy)."""

    AGAIN = 'again'
    HARD = 'hard'
    GOOD = 'good'
    EASY = 'easy'


class NodeType(StrEnum):
    """Canonical structural and semantic types for source nodes."""

    PARAGRAPH = 'paragraph'
    MATH = 'math'
    CODE = 'code'
    LIST = 'list'
    TABLE = 'table'
    IMAGE = 'image'
    CAPTION = 'caption'
    HEADER = 'header'
    BIBLIOGRAPHIC = 'bibliographic'
    NOTE = 'note'
    FOOTER = 'footer'
    MARKDOWN = 'markdown'
    INSTRUCTION = 'instruction'


@dataclass(frozen=True, slots=True)
class VisualAsset:
    """One source visual asset in a source node's document order."""

    path: str


@dataclass(slots=True)
class SourceNode:
    """One ordered unit of authoritative source evidence.

    ``content`` is the node's canonical text representation: source/OCR
    transcription for ordinary nodes and the generated visual description
    for enriched image nodes. ``assets`` preserves every attached visual
    asset in source order as the original visual evidence. Derived structure
    and knowledge records refer back to these nodes; they do not replace them.
    """

    type: NodeType | None = None
    content: str | None = None
    index: int = 0
    uuid: str | None = None
    document_index: int | None = None
    assets: list[VisualAsset] = field(default_factory=list)
    embedding: list[float] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    governing_instruction_uuids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProcedureLink:
    """Link between a source statement and procedure."""

    source: str
    statement_uuid: str
    procedure_uuid: str


class ProcedureKind(StrEnum):
    """Provenance kind for persisted procedure content."""

    SOURCE = 'source'
    GENERATED = 'generated'


@dataclass(slots=True)
class Procedure:
    """A procedure node with provenance and canonical content."""

    block: list[int]
    index: int = 0
    kind: ProcedureKind = ProcedureKind.SOURCE
    uuid: str | None = None
    statement_uuid: str | None = None
    member_positions: list[int] = field(default_factory=list)
    procedure: str | None = None

@dataclass(slots=True)
class FSRSState:
    """Current FSRS algorithm state for a card."""

    stability: float = 0.0
    difficulty: float = 0.0
    due: str | None = None  # ISO datetime
    interval: float = 0.0
    reps: int = 0
    lapses: int = 0
    last_review: str | None = None  # ISO datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            'stability': self.stability,
            'difficulty': self.difficulty,
            'due': self.due,
            'interval': self.interval,
            'reps': self.reps,
            'lapses': self.lapses,
            'last_review': self.last_review,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'FSRSState':
        return cls(
            stability=data.get('stability', 0.0),
            difficulty=data.get('difficulty', 0.0),
            due=data.get('due'),
            interval=data.get('interval', 0.0),
            reps=data.get('reps', 0),
            lapses=data.get('lapses', 0),
            last_review=data.get('last_review'),
        )


@dataclass(slots=True)
class Card:
    """A reviewable learning card attached to a hub.

    Cards are the scheduling units. One hub (EntityHub, StatementHub,
    ProcedureHub) can have multiple cards. Each card maintains its own
    FSRS state.
    """

    uuid: str
    hub_uuid: str  # The hub this card belongs to
    hub_kind: str  # 'entity' | 'statement' | 'procedure'
    prompt: str = ''
    response: str = ''
    status: CardStatus = CardStatus.ACTIVE
    fsrs: FSRSState = field(default_factory=FSRSState)
    created_at: str | None = None  # ISO datetime
    source: str | None = None  # Source key this card was generated for


@dataclass(frozen=True, slots=True)
class Review:
    """One review event for a card."""

    uuid: str
    card_uuid: str
    rating: Rating
    timestamp: str  # ISO datetime
    response_time_ms: int
    confidence: float  # 0.0 - 1.0
    session_id: str
    session_index: int
    fsrs_state_after: dict[str, Any]  # FSRSState snapshot after this review
    source: str | None = None


@dataclass(slots=True)
class Instruction:
    """An imperative instruction found in the source."""

    block: list[int]
    member_positions: list[int] = field(default_factory=list)
    uuid: str | None = None


@dataclass(slots=True)
class Statement:
    """A statement with source provenance and compiled canonical content."""

    block: list[int]
    member_positions: list[int] = field(default_factory=list)
    uuid: str | None = None
    statement: str | None = None
    instruction_uuids: list[str] = field(default_factory=list)



@dataclass(slots=True)
class Document:
    """One page-level document produced by an ingestion provider."""

    index: int
    image_path: str
    nodes: list[SourceNode] = field(default_factory=list)


@dataclass(slots=True)
class Source:
    """The complete ingested source and its ordered page documents."""

    key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ocr_response: Any = None
    documents: list[Document] = field(default_factory=list)


def source_key(source: Source | str | None) -> str | None:
    """Returns a graph-compatible source key from canonical or legacy state."""
    if isinstance(source, Source):
        return source.key
    return source


@dataclass(slots=True)
class Triplet:
    """A subject-predicate-object fact extracted from the source.

    ``occurrence_uuids`` carries the durable identity of each evidence-node
    occurrence. A triplet may have more than one evidence node, and graph
    persistence materializes one connector per occurrence.
    """

    subject: str
    predicate: str
    object: str
    evidence_positions: list[int] = field(default_factory=list)
    occurrence_uuids: dict[int, str] = field(default_factory=dict)
    entity_uuids: dict[tuple[int, str], str] = field(default_factory=dict)
    predicate_uuids: dict[int, str] = field(default_factory=dict)


class BundleValidationError(ValueError):
    """Raised when a construction bundle violates its data contracts."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__('; '.join(self.errors))


@dataclass(slots=True)
class ConstructionBundle:
    """Canonical source-local data assembled during document construction.

    The bundle is deliberately a typed collection of records and indexes, not
    a general-purpose graph.  It is safe to create one bundle per ingestion
    run because every mutable collection has an independent default.
    """

    source: Source
    nodes: list[SourceNode] = field(default_factory=list)
    instructions: list[Instruction] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    triplets: list[Triplet] = field(default_factory=list)
    knowledge_index: KnowledgeIndex | None = None
    statement_enrichment_inputs: list[StatementEnrichmentInput] = field(
        default_factory=list
    )
    procedure_enrichment_inputs: list[ProcedureEnrichmentInput] = field(
        default_factory=list
    )
    statement_hub_records: list[StatementHubRecord] = field(
        default_factory=list
    )
    procedure_hub_records: list[ProcedureHubRecord] = field(
        default_factory=list
    )
    entity_hub_bundle: HubBuildBundle | None = None
    predicate_hub_bundle: HubBuildBundle | None = None
    entity_hub_components: list[HubComponent] = field(default_factory=list)
    predicate_hub_components: list[HubComponent] = field(default_factory=list)
    triplet_memberships: list[TripletMembership] = field(default_factory=list)
    entity_hub_assignments: list[dict] = field(default_factory=list)
    predicate_hub_assignments: list[dict] = field(default_factory=list)
    entity_hub_records: list[dict] = field(default_factory=list)
    predicate_hub_records: list[dict] = field(default_factory=list)
    triplet_hubs: list[dict] = field(default_factory=list)
    statement_enrichments: list[dict] = field(default_factory=list)
    procedure_enrichments: list[dict] = field(default_factory=list)
    statement_hubs: list[dict] = field(default_factory=list)
    procedure_hubs: list[dict] = field(default_factory=list)
    generated_procedures: list[Procedure] = field(default_factory=list)
    procedure_links: list[ProcedureLink] = field(default_factory=list)
    entity_descriptions: dict[int, dict[str, str | None]] = field(
        default_factory=dict
    )
    predicate_descriptions: dict[int, dict[str, str | None]] = field(
        default_factory=dict
    )
    entity_embeddings: dict[int, dict[str, list[float]]] = field(
        default_factory=dict
    )
    predicate_embeddings: dict[int, dict[str, list[float]]] = field(
        default_factory=dict
    )
    derived: dict[str, Any] = field(default_factory=dict)
    indexes: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


def _validate_positions(
    errors: list[str], label: str, positions: list[int]
) -> None:
    """Validates a structural block without resolving it to node IDs."""
    if not positions:
        errors.append(f'{label} block must not be empty')
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            errors.append(
                f'{label} block contains non-integer position {position!r}'
            )
        elif position < 0:
            errors.append(
                f'{label} block contains negative position {position}'
            )


def _validate_members(
    errors: list[str],
    label: str,
    member_positions: list[int],
    node_ids: set[int],
) -> None:
    """Validates ordered node membership references."""
    if len(member_positions) != len(set(member_positions)):
        errors.append(f'{label} member_positions contain duplicates')
    for member in member_positions:
        if member not in node_ids:
            errors.append(f'{label} references missing node {member!r}')


def validate_bundle(
    bundle: ConstructionBundle, *, require_identities: bool = False
) -> None:
    """Validates a source-local bundle without mutating or resolving storage.

    ``require_identities`` is used at downstream boundaries after the
    construction stages have assigned durable IDs.
    """
    errors: list[str] = []
    if not bundle.source.key or not bundle.source.key.strip():
        errors.append('source key must be non-empty')

    document_indexes = [document.index for document in bundle.source.documents]
    if len(document_indexes) != len(set(document_indexes)):
        errors.append('source documents contain duplicate indexes')
    known_documents = set(document_indexes)

    num_nodes = len(bundle.nodes)
    valid_positions = set(range(num_nodes))
    for node_index, node in enumerate(bundle.nodes):
        label = f'node {node_index}'
        if (
            node.document_index is not None
            and node.document_index not in known_documents
        ):
            errors.append(
                f'{label} references missing document {node.document_index}'
            )

    records = [
        ('instruction', bundle.instructions),
        ('statement', bundle.statements),
        ('procedure', bundle.procedures),
    ]
    for label, values in records:
        for index, value in enumerate(values):
            record_label = f'{label} {index}'
            generated_procedure = (
                isinstance(value, Procedure)
                and not value.block
                and bool(value.statement_uuid)
            )
            if not generated_procedure:
                _validate_positions(errors, record_label, value.block)
            _validate_members(
                errors, record_label, value.member_positions, valid_positions
            )
            if isinstance(value, Instruction) and require_identities:
                if not value.uuid:
                    errors.append(f'{record_label} is missing a uuid')
            if isinstance(value, Statement) and require_identities:
                if not value.uuid:
                    errors.append(f'{record_label} is missing a uuid')
            if isinstance(value, Procedure) and require_identities:
                if not value.uuid:
                    errors.append(f'{record_label} is missing a uuid')
            if isinstance(value, Procedure) and value.statement_uuid == '':
                errors.append(
                    f'{record_label} statement_uuid must be non-empty'
                )

    statement_uuids = [
        statement.uuid for statement in bundle.statements if statement.uuid
    ]
    if len(statement_uuids) != len(set(statement_uuids)):
        errors.append('statements contain duplicate uuids')
    procedure_uuids = [
        procedure.uuid for procedure in bundle.procedures if procedure.uuid
    ]
    if len(procedure_uuids) != len(set(procedure_uuids)):
        errors.append('procedures contain duplicate uuids')

    # Check for duplicate node UUIDs
    node_uuids = [node.uuid for node in bundle.nodes if node.uuid]
    if len(node_uuids) != len(set(node_uuids)):
        errors.append('nodes contain duplicate uuids')

    for index, triplet in enumerate(bundle.triplets):
        label = f'triplet {index}'
        if not triplet.evidence_positions:
            errors.append(f'{label} must have evidence node positions')
        if len(triplet.evidence_positions) != len(
            set(triplet.evidence_positions)
        ):
            errors.append(f'{label} evidence contains duplicates')
        for pos in triplet.evidence_positions:
            if pos not in valid_positions:
                errors.append(f'{label} references missing node {pos!r}')
            if require_identities and not triplet.occurrence_uuids.get(pos):
                errors.append(f'{label} occurrence {pos} is missing a uuid')

    if errors:
        raise BundleValidationError(errors)


def flatten_documents(documents: list[Document]) -> list[SourceNode]:
    """Flattens canonical documents into one ordered source-evidence stream."""
    flat: list[SourceNode] = []
    for document in documents:
        for node in document.nodes or []:
            node.document_index = document.index
            flat.append(node)
    return flat
