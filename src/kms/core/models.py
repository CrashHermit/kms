"""
Domain models for the pipeline: the in-memory AST and entity data structures, the
shared AutoMathKG vocabularies, and the pure helpers that operate on them.

These are the vocabulary the whole system is *about*. They depend only on the standard
library and pydantic — deliberately free of the orchestration framework (LangGraph) and
the LLM stack (dspy), so a test, the graph tier, or a future non-LangGraph runner can use
them in isolation. The LangGraph ``state.State`` that carries these through the graph lives in
its sibling ``kms.core.state``.

The pipeline assembles an ordered AST in memory: a list of Segments (one per page, in
document order), each owning its pictures, its OCR'd markdown content, and the AST nodes
extracted from that content. The seam merger later flattens the per-page segments into one
global ordered node list (``flatten_segments``); from that point the flat list is the
single source of truth.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

# --- AST data structures ---


class NodeType(StrEnum):
    """The block-level node types the extractor may emit. Blocks only — inline
    structure (bold, inline math, links) stays inside a node's markdown content.

    The extractor is purely STRUCTURAL and domain-agnostic: it emits general document
    structure only. Semantic typing (definition / theorem / law / …) lives entirely at the
    entity layer, where it is induced per book, not here."""

    PARAGRAPH = 'paragraph'
    MATH = 'math'  # standalone display math block
    CODE = 'code'  # fenced code block
    LIST = 'list'
    TABLE = 'table'
    IMAGE = 'image'
    CAPTION = 'caption'
    HEADER = 'header'


class EntityType(StrEnum):
    """The math profile's entity types — the values the per-type finders stamp on the entities
    they emit (AutoMathKG's taxonomy). Distinct from NodeType, which is document structure.

    An entity's ``type`` is an OPEN property, not this closed set (see ``docs/GENERALIZATION.md``,
    "kind = label, type = property"): the universal attributor induces whatever a textbook calls its
    blocks — a physics ``law``, a biology ``mechanism``, a CS ``algorithm``. This enum survives as
    the *math profile's* vocabulary, used by the per-type chains that hardcode one type each; it
    constrains nothing downstream, which reads ``Entity.type`` as a plain string."""

    DEFINITION = 'definition'
    THEOREM = 'theorem'  # subsumes proposition, corollary, lemma
    PROBLEM = 'problem'  # worked examples and exercises


class ProcedureType(StrEnum):
    """The math profile's procedure types — a procedure being a named, ordered derivation
    extracted from an entity (see ``docs/UNIFIED-KG.md``). A theorem's proof and a problem's
    solution are these two values.

    Like ``EntityType`` this is a *profile* vocabulary, not a closed set: ``Procedure.type`` is an
    open string, so a physics ``derivation`` or a CS ``algorithm`` is simply another value the
    procedure finder may induce.

    A procedure's step decomposition reifies into ``:Event`` nodes; the derivation is thus the
    procedural half of the graph, distinct from the declarative ``:Entity`` it hangs off."""

    PROOF = 'proof'
    SOLUTION = 'solution'


# --- Shared AutoMathKG vocabularies (Table C4) ---
# Kept here, not in a single attributor, so every per-type attributor draws the role taxonomy
# from one source of truth instead of copying the list.

# The nine role labels ("bodylist" template), the full taxonomy across all types. Each
# attributor offers the model only the subset a given context actually exercises (e.g. a
# definition never uses proof-only roles; a theorem statement never `deduction`s). This stays
# CLOSED on purpose: it labels the internal structure of one derivation, which the AutoMathKG
# taxonomy covers well. The cross-entity relation vocabulary — once the same list — is now open
# and LLM-named (see `Reference.relation`, docs/GENERALIZATION.md step 2).
ACTIONS_ALL = [
    'premise',
    'assumption',
    'lemma',
    'corollary',
    'definition',
    'conclusion',
    'deduction',
    'calculation',
    'enumeration',
]


class BodySegment(BaseModel):
    """One `bodylist` piece: a contiguous slice of an entity's content and the role it
    plays (AutoMathKG's action label — see the per-type attributor for the allowed set).
    A pydantic model because it doubles as a DSPy structured-output type at the LLM
    boundary; stored as-is on the entity.

    `concepts` is the event-conceptualization axis (AutoSchemaKG's φ over events): when a segment
    is a *procedure step* it reifies into an `:Event`, and the conceptualizer tags it with flat
    concept phrases spanning specific → general. Empty until that stage runs (and always empty for
    a statement bodylist, which is not reified into events)."""

    description: str
    action: str
    concepts: list[str] = []


class Procedure(BaseModel):
    """One worked derivation of an entity — a proof, a solution, a physics derivation — reified by
    the graph tier into a `:Procedure` container rooting an `:Event` step chain.

    This is the unification of AutoMathKG's Thm-only `proofs` and Prob-only `solutions` (each
    `{contents, bodylist}`): one list on the entity, the flavour carried by an OPEN `type` (proof /
    solution / derivation / algorithm / …) instead of by which field it sits in — the same
    "kind general, type a property" pattern as the entity itself, one level down
    (`docs/GENERALIZATION.md`, "Entity layer").

    `contents` is the derivation's own markdown; `steps` its role-labelled decomposition (the
    paper's `bodylist`), which reifies into the `:Event` chain — a procedure with no steps is
    persisted as a bare container. `generated` marks a procedure the procedure *creator* wrote for
    a task that showed none, as opposed to one extracted from the page; refs/references_tactics
    are deferred to the graph tier, as for every derivation."""

    type: str
    contents: list[str] = []
    steps: list[BodySegment] = []
    generated: bool = False


class Reference(BaseModel):
    """One outgoing cross-entity reference — AutoMathKG's `refs` + `references_tactics` fused into a
    single record (the graph tier keeps them as one edge). `target` is the referenced entity's name
    as written ("Set", "positive definite matrix"); `kind` is what sort of thing the target is
    ("definition", "theorem", and — outside math — "law", "model", …); `relation` is how this entity
    relates to it, an OPEN, LLM-named label ("depends on", "applies", "generalizes"). Resolved to a
    graph edge onto a canonical hub keyed by (kind, normalized target), so references from different
    books/entities converge on one target.

    Both `kind` and `relation` were closed math vocabularies (`REFERENCE_KINDS`, `ACTIONS_ALL`);
    opening them is AutoSchemaKG's open-relation model, which is what lets a physics or biology book
    name its own relations instead of being forced into nine math tactics
    (`docs/GENERALIZATION.md`, step 2).

    A pydantic model like BodySegment — it doubles as a DSPy structured type at the referencer's LLM
    boundary and is carried on the entity until the graph tier turns it into an edge."""

    target: str
    kind: str  # what the target is: definition / theorem / law / model / …
    relation: str  # open, LLM-named: how this entity relates to the target


class Dependency(BaseModel):
    """One concept-level prerequisite: "you need `prerequisite` to define/prove/understand
    `dependent`" — the `(:Concept)-[:DEPENDS_ON]->(:Concept)` edge (`docs/GENERALIZATION.md`,
    "Drop :BROADER / MSC; add :DEPENDS_ON").

    It replaces the taxonomic `:BROADER` edge an MSC hierarchy would have given: a prerequisite
    answers the curriculum question a taxonomy only approximates, and unlike a taxonomy it is
    *groundable* — it is the concept-level rollup of the entity-level `:REFERENCES` graph the
    referencer already extracts. `support` is how many reference pairs grounded it, kept so the
    cycle guard can prefer the better-evidenced edge of a co-defined pair (a prerequisite graph
    must stay a DAG)."""

    dependent: str
    prerequisite: str
    support: int = 1


@dataclass(slots=True)
class Entity:
    """One pedagogical block lifted out of the node stream — a definition, a theorem, a worked
    example, a physics law — as a sparse overlay on the flat node stream (most nodes belong to no
    entity). `members` are node ids in document order: pointers back to the source nodes (persisted
    for provenance), so the later graph phase can draw edges from an entity to the chunks it came
    from. `id` is assigned when the finder overlays are flattened into the single emitted list.

    The overlays are independent and may reference the same node (members are pointers), so
    they are concatenated, not merged.

    `type` is an OPEN property, not a closed enum and not a graph label: the universal attributor
    induces it from the block's own content (definition / theorem / law / mechanism / …), so a new
    domain needs no new vocabulary (`docs/GENERALIZATION.md`). It is None until an attributor fills
    it — the block finder emits pure spans, and the per-type chains stamp their own `EntityType`.

    The self-contained AutoMathKG attributes below are filled in by the attributor pass; they stay
    unset (None / empty) until then. Three attributes come from later stages: `procedures` (the
    procedure finder's extracted/created derivations), `refs` (the referencer's cross-entity
    citations, turned into graph edges by the entity persister), and `concepts` (the
    conceptualizer's flat multi-tag conceptualization, which replaced AutoMathKG's fixed `field`)."""

    type: str | None = (
        None  # open, induced block type (definition/theorem/law/…)
    )
    members: list[int] = field(
        default_factory=list
    )  # member node ids, document order
    id: int | None = None  # assigned when overlays are flattened
    label: str | None = None  # the entity's own label, as written
    number: str | None = None  # the reference number in that label
    title: str | None = None  # short descriptive name of the concept
    contents: list[str] = field(
        default_factory=list
    )  # member markdown, a list of strings
    bodylist: list[BodySegment] = field(
        default_factory=list
    )  # role-labelled segmentation
    procedures: list[Procedure] = field(
        default_factory=list
    )  # extracted/created derivations
    refs: list[Reference] = field(
        default_factory=list
    )  # cross-entity references (referencer output)
    concepts: list[str] = field(
        default_factory=list
    )  # induced concept tags (conceptualizer output)
    instruction: str | None = None  # task-only: shared exercise-group directive


@dataclass(slots=True)
class Picture:
    """An image extracted from a page. `index` is the 1-based placeholder id that
    OCR's ![N]() markers refer to."""

    index: int
    image_path: str


@dataclass(slots=True)
class ASTNode:
    """A single extracted block node in the AST.

    Through the per-page ingestion phase (ocr, extractor, seam) a node lives inside
    its Segment. The seam merger then flattens all segments into one global ordered
    node list (see `flatten_segments`), stamping each node with a stable `id` and the
    `segment_index` of the page it came from. From that point the flat list is the single
    source of truth; `id` is how every later stage and the entity overlay reference a
    node, and `segment_index` is retained only so the assembler can resolve `![N]()`
    picture placeholders against the right page's pictures.

    `role` is a non-structural annotation the splitter may set (currently only "instruction",
    marking an exercise lead-in). It is kept off `type` deliberately: `type` is the purely
    structural taxonomy, `role` is an entity-layer hint that rides along on the node.
    """

    type: NodeType | None = None
    content: str | None = None
    id: int | None = (
        None  # stable global id, assigned once when the flat list is born
    )
    segment_index: int | None = (
        None  # originating page, for picture resolution after flattening
    )
    role: str | None = (
        None  # non-structural annotation (e.g. "instruction" lead-in), set by the splitter
    )


@dataclass(slots=True)
class Segment:
    """One page of the document. `index` is 0-based document order."""

    index: int
    image_path: str
    pictures: list[Picture] = field(default_factory=list)
    content: str | None = None  # markdown, filled by the OCR stage
    nodes: list[ASTNode] = field(
        default_factory=list
    )  # filled by the extractor stage


# --- Helpers (pure functions over the models above) ---


def merge_results_into_segments(
    segments: list[Segment], results: list[tuple[int, Any]], attr: str
) -> list[Segment]:
    """Drain a stage's ``(segment_index, value)`` reducer channel back into the ordered
    segment backbone, setting ``attr`` on each segment that has a result.

    Every map-reduce ingestion stage (corrector, extractor, seam merger) ends by folding
    its parallel workers' output back into the backbone keyed by segment index; this is
    that shared drain. Segments with no result are left untouched. Mutates and returns the
    same list (the collect steps run sequentially, so in-place is safe)."""
    by_index = dict(results)
    for segment in segments:
        if segment.index in by_index:
            setattr(segment, attr, by_index[segment.index])
    return segments


def flatten_entities(
    overlays: list[list['Entity']],
    nodes: list[ASTNode],
) -> list['Entity']:
    """Concatenate the finder overlays into one flat, document-ordered entity list and assign each
    a global id.

    Takes a list of overlays rather than three named channels because how many there are is a
    wiring choice: the per-type entity layer contributes three (problem / definition / theorem),
    the block-finder layer contributes one. The overlays are independent and may reference the same
    node more than once (members are node-id pointers) — they are concatenated, not merged.
    Ordering is by each entity's first member's position in the flat node stream; an entity with no
    members sorts to the end. Because the splitter made exercise nodes atomic upstream, a finder
    already emits one entity per exercise with distinct members, so no coarse-vs-fine
    reconciliation is needed. The assigned `id` is the entity's stable document-order position —
    the key the graph tier's entity vertex uuid is derived from — so a re-run maps onto the same
    vertices.
    """
    entities = [entity for overlay in overlays for entity in overlay]
    order = {node.id: i for i, node in enumerate(nodes)}
    big = len(order)
    entities.sort(
        key=lambda e: order.get(e.members[0], big) if e.members else big
    )
    for i, entity in enumerate(entities):
        entity.id = i
    return entities


def flatten_segments(segments: list[Segment]) -> list[ASTNode]:
    """Project the per-page segment backbone into one global ordered node list.

    Called once, by the seam merger, after page-splits are healed. Walks segments in
    document order and each segment's nodes in order, stamping every node with a stable
    monotonic `id` and its originating `segment_index`. The nodes are the same objects the
    segments hold — this assigns identity in place and returns the flat ordering. After
    this the flat list is the single source of truth; `segments[].nodes` is left as-is
    but is no longer read (picture resolution uses `segment_index`, not the node nesting).
    """
    flat: list[ASTNode] = []
    next_id = 0
    for segment in segments:
        for node in segment.nodes:
            node.id = next_id
            node.segment_index = segment.index
            next_id += 1
            flat.append(node)
    return flat
