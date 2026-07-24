"""
Domain models for the pipeline: the in-memory AST and entity data structures, the
shared AutoMathKG vocabularies, and the pure helpers that operate on them.

These are the vocabulary the whole system is *about*. They depend only on the standard
library and pydantic — deliberately free of the orchestration framework (LangGraph) and
the LLM stack (dspy), so a test, the graph tier, or a future non-LangGraph runner can use
them in isolation. The LangGraph ``State`` that carries these through the graph lives in
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
    structure only. Math-semantic typing (Definition/Theorem/Problem) lives entirely at
    the entity layer — the per-type finders — not here."""

    PARAGRAPH = "paragraph"
    MATH = "math"  # standalone display math block
    CODE = "code"  # fenced code block
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    HEADER = "header"


class EntityType(StrEnum):
    """The three math-semantic entity categories the entity finders produce
    (AutoMathKG's taxonomy). Distinct from NodeType, which is document structure."""

    DEFINITION = "definition"
    THEOREM = "theorem"  # subsumes proposition, corollary, lemma
    PROBLEM = "problem"  # worked examples and exercises


class ProcedureType(StrEnum):
    """The kinds of procedure — a named, ordered derivation attached to an entity (see
    ``docs/UNIFIED-KG.md``). A Theorem's ``proofs`` reify into ``proof`` procedures, a Problem's
    ``solutions`` into ``solution`` procedures. Generic on purpose (a physics ``derivation`` or CS
    ``algorithm`` would be more values of the same kind), but math-first the set is these two.

    Their step decomposition (a proof's ``bodylist``) reifies into ``:Event`` nodes; the derivation
    is thus the procedural half of the graph, distinct from the declarative ``:Entity`` it hangs off."""

    PROOF = "proof"
    SOLUTION = "solution"


# --- Shared AutoMathKG vocabularies (Table C4) ---
# Kept here, not in a single attributor, so every per-type attributor draws the field and
# role taxonomies from one source of truth instead of copying the lists.

# The fixed mathematical-field taxonomy ("field" template).
FIELDS = [
    "algebra",
    "geometry",
    "analysis",
    "logic",
    "probability and statistics",
    "applied mathematics",
    "foundations of mathematics",
]

# The nine role/tactic labels ("bodylist" template), the full taxonomy across all types.
# Each attributor offers the model only the subset a given context actually exercises
# (e.g. a definition never uses proof-only roles; a theorem statement never `deduction`s).
ACTIONS_ALL = [
    "premise",
    "assumption",
    "lemma",
    "corollary",
    "definition",
    "conclusion",
    "deduction",
    "calculation",
    "enumeration",
]

# The entity kinds a cross-entity reference may target (AutoMathKG's `definition:`/`theorem:`
# prefixes). Shared by the per-type referencers so the allowed set lives in one place.
REFERENCE_KINDS = ["definition", "theorem"]


class BodySegment(BaseModel):
    """One `bodylist` piece: a contiguous slice of an entity's content and the role it
    plays (AutoMathKG's action label — see the per-type attributor for the allowed set).
    A pydantic model because it doubles as a DSPy structured-output type at the LLM
    boundary; stored as-is on the entity."""

    description: str
    action: str


class Proof(BaseModel):
    """One proof of a Theorem: its own content and role-labelled decomposition (AutoMathKG's
    Thm-only `proofs`, each element `{contents, bodylist, ...}`). refs/references_tactics are
    deferred to the graph tier, so a proof reduces to contents + bodylist here. A pydantic
    model like BodySegment — it doubles as a DSPy structured type and is stored on the entity."""

    contents: list[str] = []
    bodylist: list[BodySegment] = []


class Solution(BaseModel):
    """One solution of a Problem (AutoMathKG's Prob-only `solutions`, each element
    `{contents, ...}`). A Problem carries no bodylist — even in the paper a solution's
    bodylist is empty — so a solution reduces to just its contents here."""

    contents: list[str] = []


class Reference(BaseModel):
    """One outgoing cross-entity reference — AutoMathKG's `refs` + `references_tactics` fused into a
    single record (the graph tier keeps them as one edge). `target` is the referenced entity's name
    as written ("Set", "positive definite matrix"); `kind` is its type prefix (`"definition"` /
    `"theorem"`, the paper's `definition:`/`theorem:` convention); `tactic` is the role the reference
    plays, one of `ACTIONS_ALL`. Resolved to a graph edge onto a general-entity hub keyed by
    (kind, normalized target), so references from different books/entities converge on one target.
    A pydantic model like BodySegment — it doubles as a DSPy structured type at the referencer's LLM
    boundary and is carried on the entity until the graph tier turns it into an edge."""

    target: str
    kind: str  # "definition" | "theorem"
    tactic: str  # one of ACTIONS_ALL


@dataclass(slots=True)
class Entity:
    """A math-semantic entity: a typed grouping of member nodes — a sparse overlay on the
    flat node stream (most nodes belong to no entity). `members` are node ids in document
    order: pointers back to the source nodes (persisted for provenance), so the later graph
    phase can draw edges from an entity to the chunks it came from. `id` is assigned when
    the three per-type overlays are flattened into the single emitted entity list.

    The overlays are independent and may reference the same node (members are pointers), so
    they are concatenated, not merged.

    The self-contained AutoMathKG attributes below are filled in by the per-type attributor
    passes; they stay unset (None / empty) until then. `refs` is the one cross-entity attribute —
    filled by the per-type referencer pass (after the attributor) and turned into graph edges (onto
    general-entity hubs) by the entity persister; it stays empty until the referencer runs."""

    type: EntityType
    members: list[int] = field(default_factory=list)  # member node ids, document order
    id: int | None = None  # assigned when overlays are flattened
    # Self-contained attributes (per-type attributor output). NOTE: the `field` attribute
    # (AutoMathKG's mathematical-field name) shadows `dataclasses.field` inside this class
    # body, so any attribute using `field(default_factory=...)` must be declared ABOVE it.
    label: str | None = None  # the entity's own label, as written
    number: str | None = None  # the reference number in that label
    title: str | None = None  # short descriptive name of the concept
    contents: list[str] = field(default_factory=list)  # member markdown, a list of strings
    bodylist: list[BodySegment] = field(default_factory=list)  # role-labelled segmentation
    proofs: list[Proof] = field(default_factory=list)  # Theorem-only: its proof(s)
    solutions: list[Solution] = field(default_factory=list)  # Problem-only: its solution(s)
    refs: list[Reference] = field(
        default_factory=list
    )  # cross-entity references (referencer output)
    field: str | None = None  # mathematical field (fixed taxonomy)
    instruction: str | None = None  # Problem-only: shared exercise-group directive


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
    `seg_index` of the page it came from. From that point the flat list is the single
    source of truth; `id` is how every later stage and the entity overlay reference a
    node, and `seg_index` is retained only so the assembler can resolve `![N]()`
    picture placeholders against the right page's pictures.

    `role` is a non-structural annotation the splitter may set (currently only "instruction",
    marking an exercise lead-in). It is kept off `type` deliberately: `type` is the purely
    structural taxonomy, `role` is an entity-layer hint that rides along on the node.
    """

    type: NodeType | None = None
    content: str | None = None
    id: int | None = None  # stable global id, assigned once when the flat list is born
    seg_index: int | None = None  # originating page, for picture resolution after flattening
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
    nodes: list[ASTNode] = field(default_factory=list)  # filled by the extractor stage


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
    problem: list["Entity"],
    definition: list["Entity"],
    theorem: list["Entity"],
    nodes: list[ASTNode],
) -> list["Entity"]:
    """Concatenate the three per-type finder overlays into one flat, document-ordered entity list
    and assign each a global id.

    The overlays are independent and may reference the same node more than once (members are
    node-id pointers) — they are concatenated, not merged. Ordering is by each entity's first
    member's position in the flat node stream; an entity with no members sorts to the end. Because
    the splitter made exercise nodes atomic upstream, the problem finder already emits one entity
    per exercise with distinct members, so no coarse-vs-fine reconciliation is needed. The assigned
    `id` is the entity's stable document-order position — the key the graph tier's entity vertex
    uuid is derived from — so a re-run maps onto the same vertices.
    """
    entities = list(problem) + list(definition) + list(theorem)
    order = {node.id: i for i, node in enumerate(nodes)}
    big = len(order)
    entities.sort(key=lambda e: order.get(e.members[0], big) if e.members else big)
    for i, entity in enumerate(entities):
        entity.id = i
    return entities


def flatten_segments(segments: list[Segment]) -> list[ASTNode]:
    """Project the per-page segment backbone into one global ordered node list.

    Called once, by the seam merger, after page-splits are healed. Walks segments in
    document order and each segment's nodes in order, stamping every node with a stable
    monotonic `id` and its originating `seg_index`. The nodes are the same objects the
    segments hold — this assigns identity in place and returns the flat ordering. After
    this the flat list is the single source of truth; `segments[].nodes` is left as-is
    but is no longer read (picture resolution uses `seg_index`, not the node nesting).
    """
    flat: list[ASTNode] = []
    next_id = 0
    for segment in segments:
        for node in segment.nodes:
            node.id = next_id
            node.seg_index = segment.index
            next_id += 1
            flat.append(node)
    return flat
