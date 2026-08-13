from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ASTNode:
    type: str | None = None
    content: str | None = None
    id: int | None = None
    segment_index: int | None = None
    image_path: str | None = None


@dataclass(slots=True)
class Step:
    text: str
    index: int = 0


@dataclass(slots=True)
class Procedure:
    block: list[int]
    index: int = 0
    members: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)


@dataclass(slots=True)
class Instruction:
    block: list[int]
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    block: list[int]
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Picture:
    index: int
    image_path: str


@dataclass(slots=True)
class Segment:
    index: int
    image_path: str
    pictures: list[Picture] = field(default_factory=list)
    content: str | None = None
    nodes: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class Triplet:
    subject: str
    predicate: str
    object: str
    node_ids: list[int] = field(default_factory=list)


def merge_results_into_segments(
    segments: list[Segment], results: list[tuple[int, Any]], attr: str
) -> list[Segment]:
    by_index = dict(results)
    for segment in segments:
        if segment.index in by_index:
            setattr(segment, attr, by_index[segment.index])
    return segments


def flatten_segments(segments: list[Segment]) -> list[ASTNode]:
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
