"""Core dataclasses shared across the ingestion pipeline."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from kms.core import content


@dataclass(frozen=True, slots=True)
class StatementEnrichmentInput:
    """Complete typed input for one statement enrichment operation."""

    statement_uuid: str
    statement: content.Content
    canonical_knowledge: str


@dataclass(frozen=True, slots=True)
class ProcedureEnrichmentInput:
    """Complete typed input for one procedure enrichment operation."""

    procedure_uuid: str
    statement: content.Content
    procedure: content.Content
    canonical_knowledge: str


@dataclass(frozen=True, slots=True)
class ProcedureMaterializationInput:
    """Source-scoped input for statement-centered procedure materialization."""

    source: str
    statement_uuid: str
    statement: content.Content
    procedure_uuid: str | None = None
    procedure: content.Content | None = None
    entity_definitions: str = ''
    has_steps: bool = False
    member_count: int = 0


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


@dataclass(slots=True)
class Node:
    """One ordered canonical node within a source document."""

    type: NodeType | None = None
    content: str | None = None
    index: int = 0
    id: int | None = None
    document_index: int | None = None
    image_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Step:
    """One numbered step inside a Procedure."""

    text: str
    index: int = 0


@dataclass(frozen=True, slots=True)
class ProcedureStepUpdate:
    """Generated ordered steps for an existing procedure."""

    source: str
    procedure_uuid: str
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class ProcedureLink:
    """Link between a source statement and procedure."""

    source: str
    statement_uuid: str
    procedure_uuid: str


@dataclass(slots=True)
class Procedure:
    """A procedure found in the source, with its member node ids."""

    block: list[int]
    index: int = 0
    uuid: str | None = None
    statement_uuid: str | None = None
    members: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)


@dataclass(slots=True)
class Instruction:
    """An imperative instruction found in the source."""

    block: list[int]
    members: list[int] = field(default_factory=list)
    uuid: str | None = None


@dataclass(slots=True)
class Statement:
    """A declarative statement found in the source."""

    block: list[int]
    members: list[int] = field(default_factory=list)
    uuid: str | None = None


@dataclass(slots=True)
class Picture:
    """A picture referenced by a segment, keyed by its local index."""

    index: int
    image_path: str


@dataclass(slots=True)
class Document:
    """One page-level document produced by an ingestion provider."""

    index: int
    image_path: str
    content: str | None = None
    pictures: list[Picture] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    node_ids: list[int] = field(default_factory=list)
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
    nodes: list[Node] = field(default_factory=list)
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
    procedure_materialization_inputs: list[ProcedureMaterializationInput] = field(
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
    procedure_step_updates: list[ProcedureStepUpdate] = field(default_factory=list)
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
    errors: list[str], label: str, members: list[int], node_ids: set[int]
) -> None:
    """Validates ordered node membership references."""
    if len(members) != len(set(members)):
        errors.append(f'{label} members contain duplicates')
    for member in members:
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

    node_ids: set[int] = set()
    for node_index, node in enumerate(bundle.nodes):
        label = f'node {node_index}'
        if node.id is None:
            errors.append(f'{label} is missing an id')
        elif node.id in node_ids:
            errors.append(f'{label} duplicates node id {node.id}')
        else:
            node_ids.add(node.id)
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
            _validate_members(errors, record_label, value.members, node_ids)
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

    for index, triplet in enumerate(bundle.triplets):
        label = f'triplet {index}'
        if not triplet.node_ids:
            errors.append(f'{label} must have evidence node ids')
        if len(triplet.node_ids) != len(set(triplet.node_ids)):
            errors.append(f'{label} evidence contains duplicates')
        for node_id in triplet.node_ids:
            if node_id not in node_ids:
                errors.append(f'{label} references missing node {node_id!r}')
            if require_identities and not triplet.occurrence_uuids.get(node_id):
                errors.append(
                    f'{label} occurrence {node_id} is missing a uuid'
                )

    if errors:
        raise BundleValidationError(errors)


def flatten_documents(documents: list[Document]) -> list[Node]:
    """Flattens canonical documents into one id-ordered node stream."""
    flat: list[Node] = []
    for document in documents:
        pictures = list(document.pictures or [])
        picture_cursor = 0
        for node in document.nodes or []:
            node.document_index = document.index
            if node.type == NodeType.IMAGE and picture_cursor < len(pictures):
                node.image_path = pictures[picture_cursor].image_path
                picture_cursor += 1
            flat.append(node)
    for index, node in enumerate(flat):
        node.id = index
    return flat
