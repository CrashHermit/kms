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
    structure (bold, inline math, links) stays inside a node's markdown content."""
    PARAGRAPH = "paragraph"
    MATH = "math"                # standalone display math block
    CODE = "code"                # fenced code block
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    HEADER = "header"
    INSTRUCTION = "instruction"  # math-book specific: shared lead for a group of problems
    PROBLEM = "problem"          # math-book specific: a single student problem to solve (exercise or worked example)


class EntityType(StrEnum):
    """The three math-semantic entity categories the grouping layer produces
    (AutoMathKG's taxonomy). Distinct from NodeType, which is document structure."""
    DEFINITION = "definition"
    THEOREM = "theorem"      # subsumes proposition, corollary, lemma
    PROBLEM = "problem"      # atomic exercises (wrapped 1:1) and gathered worked examples


class EntityRole(StrEnum):
    """The role a member node plays within its entity — the attribute the Stage 2
    pass assigns (AutoMathKG's bodylist action labels)."""
    STATEMENT = "statement"  # the definition/theorem/problem statement (all types)
    PROOF = "proof"          # theorem only, repeatable
    SOLUTION = "solution"    # problem only, repeatable


@dataclass
class Member:
    """One member node of an entity, with the role it plays. `role` is None until the
    Stage 2 attribute pass fills it."""
    node_id: int
    role: EntityRole | None = None


@dataclass
class Entity:
    """A math-semantic entity: a typed grouping that references its member nodes by
    id, a sparse overlay on the flat node stream (most nodes belong to no entity).

    `members` carry roles once the Stage 2 attribute pass runs; `number`/`instruction`
    are lifted there from the member nodes (problems only). `tail_open`/
    `head_continuation` are transient reconciliation flags set by the per-window worker
    for entities at a window edge; the grouper's collect clears them once cross-window
    continuations have been stitched."""
    type: EntityType
    members: list[Member] = field(default_factory=list)  # member nodes in document order
    id: int | None = None                                # assigned in collect, document order
    number: str | None = None                            # lifted from the member node (problems)
    instruction: str | None = None                       # lifted governing lead (problems)
    tail_open: bool = False                              # last run hit window end still gathering
    head_continuation: bool = False                     # first run continues an entity from a prior window


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
    number: str | None = None          # problem label, filled by the problem refiner (metadata only)
    instruction: str | None = None     # governing lead text, filled by the instruction governor (problems only)
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
    per-page nodes into `nodes` — the global ordered node list that every refinement stage
    after it (problem_refiner, governor, entity grouping) works on. `segments` is retained
    past that point only for its pictures (picture resolution at assembly). Both backbones
    use the default overwrite reducer because only the sequential collect steps write them.

    The `*_results` channels are map-reduce scratch space: parallel Send workers append
    entries and the stage's collect step drains them back into the active backbone. They
    carry an operator.add reducer so concurrent worker writes merge rather than clash.
    Ingestion channels key by segment index; post-flatten channels key by node id.
    """
    segments: list[Segment]
    nodes: list[ASTNode]
    entities: list[Entity]
    correction_results: Annotated[list[tuple[int, str]], operator.add]  # (segment index, corrected markdown)
    extract_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_even_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_odd_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    problem_results: Annotated[list[tuple[int, str | None]], operator.add]      # (node id, number)
    governance_results: Annotated[list[tuple[int, str]], operator.add]          # (node id, instruction)
    entity_results: Annotated[list[tuple[int, list[Entity]]], operator.add]     # (window index, entities)
    attribute_results: Annotated[list[tuple[int, list[str]]], operator.add]     # (entity id, role per member)


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
