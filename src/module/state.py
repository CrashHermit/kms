"""
Shared in-memory pipeline state for the processing nodes.

The pipeline assembles an ordered AST in memory: a list of Segments (one per page,
in document order), each owning its pictures, its OCR'd markdown content, and the
AST nodes extracted from that content. `segments` is the ordered backbone and the
single source of truth.

Parallel Send workers never mutate `segments` directly. Each worker returns a
`(segment_index, result)` entry into a per-stage reducer channel (operator.add, so
concurrent writes merge instead of clashing), and the stage's `collect` step drains
that channel back into the matching Segment keyed by its index. Because the merge
keys into the already-ordered backbone, document order is preserved without a
separate sort — the reducer's arrival order does not matter.
"""

import base64
import operator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypedDict

import dspy
from pydantic import BaseModel


# --- AST data structures ---

class NodeType(StrEnum):
    """The block-level node types the extractor may emit. Blocks only — inline
    structure (bold, inline math, links) stays inside a node's markdown content.

    The extractor is purely STRUCTURAL and domain-agnostic: it emits general document
    structure only. Math-semantic typing (Definition/Theorem/Problem) lives entirely at
    the entity layer — the per-type finders — not here."""
    PARAGRAPH = "paragraph"
    MATH = "math"                # standalone display math block
    CODE = "code"                # fenced code block
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    HEADER = "header"


class EntityType(StrEnum):
    """The three math-semantic entity categories the entity finders produce
    (AutoMathKG's taxonomy). Distinct from NodeType, which is document structure."""
    DEFINITION = "definition"
    THEOREM = "theorem"      # subsumes proposition, corollary, lemma
    PROBLEM = "problem"      # worked examples and exercises


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
    bodylist is empty — and refs/references_tactics are deferred to the graph tier, so a
    solution reduces to just its contents here."""
    contents: list[str] = []


@dataclass
class Entity:
    """A math-semantic entity: a typed grouping of member nodes — a sparse overlay on the
    flat node stream (most nodes belong to no entity). `members` are node ids in document
    order: pointers back to the source nodes (persisted for provenance), so the later graph
    phase can draw edges from an entity to the chunks it came from. `id` is assigned when
    the three per-type overlays are flattened into the single emitted entity list.

    The overlays are independent and may reference the same node (members are pointers), so
    they are concatenated, not merged.

    The self-contained AutoMathKG attributes below are filled in by the per-type attributor
    passes (only the Definition attributor exists so far); they stay unset (None / empty)
    until then. Cross-entity attributes (refs / references_tactics) and the type-specific
    proofs/solutions are not here yet — they belong to the later graph tier."""
    type: EntityType
    members: list[int] = field(default_factory=list)  # member node ids, document order
    id: int | None = None                             # assigned when overlays are flattened
    # Self-contained attributes (per-type attributor output). NOTE: the `field` attribute
    # (AutoMathKG's mathematical-field name) shadows `dataclasses.field` inside this class
    # body, so any attribute using `field(default_factory=...)` must be declared ABOVE it.
    label: str | None = None                          # the entity's own label, as written
    number: str | None = None                         # the reference number in that label
    title: str | None = None                          # short descriptive name of the concept
    contents: list[str] = field(default_factory=list) # member markdown, a list of strings
    bodylist: list[BodySegment] = field(default_factory=list)  # role-labelled segmentation
    proofs: list[Proof] = field(default_factory=list) # Theorem-only: its proof(s)
    solutions: list[Solution] = field(default_factory=list)  # Problem-only: its solution(s)
    field: str | None = None                          # mathematical field (fixed taxonomy)
    instruction: str | None = None                    # Problem-only: shared exercise-group directive


@dataclass
class Picture:
    """An image extracted from a page. `index` is the 1-based placeholder id that
    OCR's ![N]() markers refer to."""
    index: int
    image_path: str


@dataclass
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
    id: int | None = None              # stable global id, assigned once when the flat list is born
    seg_index: int | None = None       # originating page, for picture resolution after flattening
    role: str | None = None            # non-structural annotation (e.g. "instruction" lead-in), set by the splitter


@dataclass
class Segment:
    """One page of the document. `index` is 0-based document order."""
    index: int
    image_path: str
    pictures: list[Picture] = field(default_factory=list)
    content: str | None = None                          # markdown, filled by the OCR stage
    nodes: list[ASTNode] = field(default_factory=list)  # filled by the extractor stage


# --- LangGraph state ---

class State(TypedDict, total=False):
    """Shared state for every stage.

    Two backbones, one after the other. During per-page ingestion (corrector, extractor,
    seam) `segments` is the ordered backbone. The seam merger then flattens the healed
    per-page nodes into `nodes` — the global ordered node list that the per-type entity
    finders each walk. `segments` is retained past that point only for its pictures
    (picture resolution at assembly). Both backbones use the default overwrite reducer
    because only the sequential collect steps write them.

    The three `*_entities` channels are each written once, by their own finder (the
    finders run in parallel over `nodes`). They are independent sparse overlays and may
    reference the same node from more than one entity — that is fine, members are node-id
    pointers. Because the splitter has already made exercise nodes atomic (one node per
    exercise), the problem finder emits one entity per exercise with distinct members — no
    coarse-vs-fine reconciliation is needed. `run()` concatenates the three into one flat,
    document-ordered entity list with global ids for the emitted `entities.json`.

    The `*_results` channels are map-reduce scratch space: parallel Send workers append
    entries and the stage's collect step drains them back into the active backbone. They
    carry an operator.add reducer so concurrent worker writes merge rather than clash.
    """
    segments: list[Segment]
    nodes: list[ASTNode]
    problem_entities: list[Entity]      # written by the problem finder
    definition_entities: list[Entity]   # written by the definition finder
    theorem_entities: list[Entity]      # written by the theorem finder
    correction_results: Annotated[list[tuple[int, str]], operator.add]  # (segment index, corrected markdown)
    extract_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_even_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_odd_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]


# --- Helpers ---

def load_dspy_image(path: str | None) -> dspy.Image | None:
    """Load a PNG from disk into a dspy.Image (base64 data URL)."""
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return dspy.Image(url=f"data:image/png;base64,{encoded}")


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
