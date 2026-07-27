"""
Domain models for the pipeline: the in-memory AST and entity data structures, and the
pure helpers that operate on them.

These are the vocabulary the whole system is *about*. They depend only on the standard
library — deliberately free of the orchestration framework (LangGraph) and the LLM stack
(dspy), so a test, the graph tier, or a future non-LangGraph runner can use them in
isolation. The LangGraph ``state.State`` that carries these through the graph lives in its
sibling ``kms.core.state``.

The pipeline assembles an ordered AST in memory: a list of Segments (one per page, in
document order), each owning its pictures, its OCR'd markdown content, and the AST nodes
extracted from that content. The seam merger later flattens the per-page segments into one
global ordered node list (``flatten_segments``); from that point the flat list is the
single source of truth.

Persisted vs. transient (see ``docs/SCHEMA.md``): these models are the pipeline's *working
state*, and the graph is the deliverable. A field is persisted only if something reads it
back from the graph; fields that exist purely to pass information between stages are
transient and are labelled as such below (currently all of ``Segment``).
``Segment``).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --- AST data structures ---


class NodeType(StrEnum):
    """The block-level node types the extractor may emit. Blocks only — inline
    structure (bold, inline math, links) stays inside a node's markdown content.

    The extractor is purely STRUCTURAL and domain-agnostic: it emits general document
    structure only. What *kind* of pedagogical block a run of nodes forms (definition,
    theorem, example, law, …) lives entirely at the entity layer — the statement
    extractor's open ``Entity.type`` — not here."""

    PARAGRAPH = 'paragraph'
    MATH = 'math'  # standalone display math block
    CODE = 'code'  # fenced code block
    LIST = 'list'
    TABLE = 'table'
    IMAGE = 'image'
    CAPTION = 'caption'
    HEADER = 'header'
    INSTRUCTION = 'instruction'  # exercise lead-in, set by the instruction finder
    STATEMENT = 'statement'  # placeholder statement node, set when a procedure is orphaned
    PROCEDURE = 'procedure'  # placeholder procedure node, set when a statement lacks a derivation


@dataclass(slots=True)
class Procedure:
    """One worked derivation of an entity — a proof, a solution, a derivation.

    Found as its own span by the pedagogical component finder (so it carries ``members``, its own node
    ids, and gets real ``:DERIVED_FROM`` provenance) and decomposed into ordered ``steps``
    by the procedure extractor. Decomposition is universal: every procedure decomposes,
    whatever it derives.

    ``steps`` are VERBATIM slices that partition ``contents`` — concatenating them in order
    reproduces the source with nothing added or removed. They reify into the graph's
    ``:Act`` chain.

    Deliberately no ``type`` (proof / solution / derivation): it is derivable from the
    owning entity's ``type``, so storing it would duplicate a neighbour's fact (see
    ``docs/SCHEMA.md``, principle 5). ``index`` orders an entity's procedures — a theorem
    with two proofs owns two of these."""

    index: int = 0
    members: list[int] = field(
        default_factory=list
    )  # member node ids, document order
    contents: list[str] = field(default_factory=list)  # member markdown
    steps: list[str] = field(
        default_factory=list
    )  # verbatim partition of contents


@dataclass(slots=True)
class Entity:
    """A pedagogical block: a typed grouping of member nodes — a sparse overlay on the
    flat node stream (most nodes belong to no entity).

    MACRO ONLY. This is a document region — a theorem statement, a definition, a worked
    example, an exercise — not the fine-grained noun-phrase entity of AutoSchemaKG (that
    kind does not exist in this schema). ``members`` are node ids in document order:
    pointers back to the source nodes, so the graph phase can draw provenance edges from
    an entity to the chunks it came from. ``id`` is assigned when the finder's overlay is
    flattened into the emitted entity list.

    ``type`` is an OPEN, induced string (``definition`` / ``theorem`` / ``example`` /
    ``law`` / …) filled by the statement extractor — never a closed enum, and never a
    Neo4j label (open label sets explode). It records the block's *genre*, which concept
    tags cannot recover: a definition of X and an exercise about X carry identical
    concepts and are entirely different objects.

    The attributes below stay unset until the statement extractor fills them.
    ``procedures`` is filled by the procedure extractor, which runs after."""

    members: list[int] = field(
        default_factory=list
    )  # member node ids, document order
    id: int | None = None  # assigned when the overlay is flattened
    type: str | None = None  # open, induced block genre
    label: str | None = None  # the block's own label, as written
    number: str | None = None  # the reference number in that label
    title: str | None = None  # short descriptive name of the concept
    contents: list[str] = field(
        default_factory=list
    )  # member markdown, a list of strings
    procedures: list[Procedure] = field(
        default_factory=list
    )  # worked derivations (procedure extractor output)
    instruction: str | None = (
        None  # shared exercise-group directive (instruction distributor output)
    )


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

    `role` was TRANSIENT — pipeline state, never persisted. The instruction finder stamped
    "instruction" on exercise lead-in nodes; the group finder read it to treat those nodes
    as hard boundaries, and the instruction distributor read it to find the lead-ins whose
    directive it propagates. The instruction finder is retired and lead-ins are now their own
    spans from the pedagogical component finder, so the field is removed.
    """

    type: NodeType | None = None
    content: str | None = None
    id: int | None = (
        None  # stable global id, assigned once when the flat list is born
    )
    segment_index: int | None = (
        None  # originating page, for picture resolution after flattening
    )


@dataclass(slots=True)
class Segment:
    """One page of the document. `index` is 0-based document order.

    TRANSIENT — per-page ingestion scaffolding. Only its pictures outlive the seam merger,
    and only for placeholder resolution at assembly; no Segment reaches the graph."""

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
    entities: list['Entity'], nodes: list[ASTNode]
) -> list['Entity']:
    """Order the finder's entity overlay by document position and assign each a global id.

    Ordering is by each entity's first member's position in the flat node stream; an entity
    with no members sorts to the end. One finder produces one partition — entities no longer
    overlap, so there is nothing to merge or reconcile, only to order. The assigned ``id`` is
    the entity's stable document-order position — the key the graph tier's entity vertex uuid
    is derived from — so a re-run maps onto the same vertices.

    Args:
        entities: The block finder's overlay, in any order.
        nodes: The flat node stream, for resolving member positions.

    Returns:
        The same entities, sorted into document order with ``id`` assigned.
    """
    ordered = list(entities)
    order = {node.id: i for i, node in enumerate(nodes)}
    last = len(order)
    ordered.sort(
        key=lambda e: order.get(e.members[0], last) if e.members else last
    )
    for i, entity in enumerate(ordered):
        entity.id = i
    return ordered


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
