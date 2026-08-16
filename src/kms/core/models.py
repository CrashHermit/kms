"""Core dataclasses shared across the ingestion pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ASTNode:
    """One node of the flattened document AST.

    Nodes carry positional, content, and typing fields; ids are stable
    across pipeline stages once assigned by ``flatten_segments``.
    """

    type: str | None = None
    content: str | None = None
    id: int | None = None
    segment_index: int | None = None
    image_path: str | None = None


@dataclass(slots=True)
class Step:
    """One numbered step inside a Procedure."""

    text: str
    index: int = 0


@dataclass(slots=True)
class Procedure:
    """A procedure found in the source, with its member node ids."""

    block: list[int]
    index: int = 0
    statement_uuid: str | None = None
    members: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)


@dataclass(slots=True)
class Instruction:
    """An imperative instruction found in the source."""

    block: list[int]
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    """A declarative statement found in the source."""

    block: list[int]
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Picture:
    """A picture referenced by a segment, keyed by its local index."""

    index: int
    image_path: str


@dataclass(slots=True)
class Segment:
    """One OCR'd page of a source document.

    Holds the raw transcription, any pictures on the page, and the
    AST nodes later derived from the content.
    """

    index: int
    image_path: str
    pictures: list[Picture] = field(default_factory=list)
    content: str | None = None
    nodes: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class Triplet:
    """A subject-predicate-object fact extracted from the source."""

    subject: str
    predicate: str
    object: str
    node_ids: list[int] = field(default_factory=list)


def merge_results_into_segments(
    segments: list[Segment], results: list[tuple[int, Any]], attr: str
) -> list[Segment]:
    """Writes per-segment pipeline results back onto the segments.

    Args:
        segments: The segments to update, in place.
        results: ``(segment_index, value)`` pairs from a worker pass.
        attr: The segment attribute to set for each result.

    Returns:
        The same segments list, mutated.
    """
    by_index = dict(results)
    for segment in segments:
        if segment.index in by_index:
            setattr(segment, attr, by_index[segment.index])
    return segments


def flatten_segments(segments: list[Segment]) -> list[ASTNode]:
    """Flattens all segments into one id-ordered AST node stream.

    Picture records are reattached to their image nodes by local index,
    then every node gets a stable sequential id across the whole source.

    Args:
        segments: The segments whose nodes are flattened.

    Returns:
        A single list of nodes ordered by segment then position.
    """
    flat: list[ASTNode] = []
    for segment in segments:
        pictures = list(segment.pictures or [])
        picture_cursor = 0
        for node in segment.nodes or []:
            node.segment_index = segment.index
            if node.type == 'image' and picture_cursor < len(pictures):
                node.image_path = pictures[picture_cursor].image_path
                picture_cursor += 1
            flat.append(node)
    for index, node in enumerate(flat):
        node.id = index
    return flat
