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
    INSTRUCTION = "instruction"  # math-book specific: shared lead for a group of exercises
    EXERCISE = "exercise"        # math-book specific: a single student problem to solve


@dataclass
class Picture:
    """An image extracted from a page. `index` is the 1-based placeholder id that
    OCR's ![N]() markers refer to."""
    index: int
    image_path: str


@dataclass
class ASTNode:
    """A single extracted block node in the AST."""
    type: NodeType | None = None
    content: str | None = None
    number: str | None = None                                   # exercise label, filled by the exercise refiner
    exercise_numbers: list[str] = field(default_factory=list)   # flat exercise labels an instruction governs, filled by the instruction refiner


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

    `segments` is the ordered AST backbone; only the sequential collect steps write
    it, so the default overwrite reducer is correct. The `*_results` channels are
    map-reduce scratch space: parallel Send workers append one entry per segment and
    the stage's collect step drains them back into `segments`. They carry an
    operator.add reducer so concurrent worker writes merge rather than clash.
    """
    segments: list[Segment]
    filter_results: Annotated[list[tuple[int, list[Picture]]], operator.add]
    ocr_results: Annotated[list[tuple[int, str]], operator.add]
    extract_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    exercise_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    instruction_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_even_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_odd_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]


# --- Helpers ---

def load_dspy_image(path: str | None) -> dspy.Image | None:
    """Load a PNG from disk into a dspy.Image (base64 data URL, mirroring Paideia)."""
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return dspy.Image(url=f"data:image/png;base64,{encoded}")


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
            for pic_path in sorted(images_dir.glob("Image_*.png")):
                file_no = int(pic_path.stem.split("_")[1])
                # picture_extractor saved pictures 0-based; the OCR placeholder id is 1-based.
                pictures.append(Picture(index=file_no + 1, image_path=str(pic_path)))
        segments.append(Segment(
            index=index,
            image_path=str(seg_dir / "Segment.png"),
            pictures=pictures,
        ))
    return segments
