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


@dataclass
class Entity:
    """A math-semantic entity: a typed grouping of member nodes — a sparse overlay on the
    flat node stream (most nodes belong to no entity). `members` are node ids in document
    order: pointers back to the source nodes (persisted for provenance), so the later graph
    phase can draw edges from an entity to the chunks it came from. `id` is assigned when
    the three per-type overlays are flattened into the single emitted entity list.

    The overlays are independent and may reference the same node (members are pointers), so
    they are concatenated, not merged. Per-attribute detail (member roles, number,
    instruction, …) is added by later per-attribute passes that do not exist yet, so an
    entity is just `{id, type, members}` for now."""
    type: EntityType
    members: list[int] = field(default_factory=list)  # member node ids, document order
    id: int | None = None                             # assigned when overlays are flattened


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
    """
    type: NodeType | None = None
    content: str | None = None
    id: int | None = None              # stable global id, assigned once when the flat list is born
    seg_index: int | None = None       # originating page, for picture resolution after flattening


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
    pointers. `run()` concatenates the three into one flat, document-ordered entity list
    with global ids for the emitted `entities.json`.

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
    """Load a PNG from disk into a dspy.Image (base64 data URL, mirroring Paideia)."""
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
