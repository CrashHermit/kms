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

    Two backbones, one after the other. During per-page ingestion (image_filter, ocr,
    extractor, seam) `segments` is the ordered backbone. The seam merger then flattens
    the healed per-page nodes into `nodes` — the global ordered node list that every
    refinement stage after it (problem_refiner, governor, entity grouping) works on.
    `segments` is retained past that point only for its pictures (picture resolution
    at assembly). Both backbones use the default overwrite reducer because only the
    sequential collect steps write them.

    The `*_results` channels are map-reduce scratch space: parallel Send workers append
    entries and the stage's collect step drains them back into the active backbone. They
    carry an operator.add reducer so concurrent worker writes merge rather than clash.
    Ingestion channels key by segment index; post-flatten channels key by node id.
    """
    segments: list[Segment]
    nodes: list[ASTNode]
    filter_results: Annotated[list[tuple[int, list[Picture]]], operator.add]
    ocr_results: Annotated[list[tuple[int, str]], operator.add]
    extract_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_even_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_odd_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    problem_results: Annotated[list[tuple[int, str | None]], operator.add]      # (node id, number)
    governance_results: Annotated[list[tuple[int, str]], operator.add]          # (node id, instruction)


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


def load_segments(output_dir: str | Path = "output") -> list[Segment]:
    """Build the initial ordered Segment list from the picture_extractor disk tree.

    Expects the layout written by picture_extractor.extract:
        <output_dir>/Segments/Segment_XXXX/Segment.png
        <output_dir>/Segments/Segment_XXXX/Images/Image_YYY.png
    """
    segments_dir = Path(output_dir) / "Segments"
    segments: list[Segment] = []
    for seg_dir in sorted(segments_dir.glob("Segment_*")):
        index = int(seg_dir.name.split("_")[1])
        pictures: list[Picture] = []
        images_dir = seg_dir / "Images"
        if images_dir.is_dir():
            # Provisional 1-based numbering by reading order. The image filter
            # re-establishes a contiguous 1..N over the survivors once noise
            # pictures are dropped, so this initial id is only a placeholder.
            for pic_no, pic_path in enumerate(sorted(images_dir.glob("Image_*.png")), start=1):
                pictures.append(Picture(index=pic_no, image_path=str(pic_path)))
        segments.append(Segment(
            index=index,
            image_path=str(seg_dir / "Segment.png"),
            pictures=pictures,
        ))
    return segments
