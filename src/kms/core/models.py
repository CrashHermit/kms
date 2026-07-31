"""
Domain models for the pipeline: the in-memory AST and entity data structures,
and the pure helpers that operate on them.

These are the vocabulary the whole system is *about*. They depend only on the
standard library — deliberately free of the orchestration framework (LangGraph)
and the LLM stack (dspy), so a test, the graph tier, or a future non-LangGraph
runner can use them in isolation. The LangGraph ``state.State`` that carries
these through the graph lives in its sibling ``kms.core.state``.
"""

from dataclasses import dataclass, field
from typing import Any

# --- AST node types ---------------------------------------------------------


@dataclass(slots=True)
class ASTNode:
    """Base node in the flat document stream.

    Subclassed by structural kind (paragraph, math, code, …).
    """

    content: str | None = None
    id: int | None = None
    segment_index: int | None = None

    @property
    def kind(self) -> str:
        """Lowercase class name — the string the old NodeType enum held."""
        return type(self).__name__.removesuffix('Node').lower()


@dataclass(slots=True)
class ParagraphNode(ASTNode):
    """Prose text. Inline math stays in the paragraph."""


@dataclass(slots=True)
class MathNode(ASTNode):
    """A standalone display-math block."""


@dataclass(slots=True)
class CodeNode(ASTNode):
    """A fenced code block."""


@dataclass(slots=True)
class ListNode(ASTNode):
    """A bullet or numbered list, kept whole."""


@dataclass(slots=True)
class TableNode(ASTNode):
    """A markdown table body (grid rows only)."""


@dataclass(slots=True)
class ImageNode(ASTNode):
    """An indexed figure placeholder, ``![N]()``."""


@dataclass(slots=True)
class CaptionNode(ASTNode):
    """A figure caption, table title, note, or label."""


@dataclass(slots=True)
class HeaderNode(ASTNode):
    """A section, chapter, or block heading."""


@dataclass(slots=True)
class BibliographicNode(ASTNode):
    """One bibliographic reference to an external work.

    A footnote citation or a single entry in a reference list — a work the
    document points at rather than something the document says. Content only,
    like every other structural node; parsing the entry into authors/year/
    title/venue is a later concern.
    """


@dataclass(slots=True)
class NoteNode(ASTNode):
    """One authorial note bound to the body by a reference marker.

    A footnote, endnote, or margin note: printed outside the running text but
    saying something about the subject, unlike page furniture. Kept in the
    stream and still eligible for the semantic chain — a footnote that defines
    a term is a definition wherever it is printed. Its marker survives in
    ``content``; resolving that marker back to the body node it annotates is a
    later concern.
    """


@dataclass(slots=True)
class InstructionNode(ASTNode):
    """Exercise lead-in, set by the instruction finder."""

    pass


# --- Semantic overlay -------------------------------------------------------
#
# The overlay sits BESIDE the node stream, never in it. A node is one verbatim
# block of the page; a Statement is the whole group of blocks a pedagogical
# unit occupies, and its content is derived from them. Deliberately NOT an
# ASTNode: while it was one, a Statement could be — and was — assigned into the
# stream in its first member's place, which made every consumer of that stream
# read the group's text twice (the duplicated blocks in the assembled output).
#
# Statement and Procedure share no base class either. Their only common field
# is ``content``; their identities differ, and a Procedure is a Statement's
# CHILD rather than its sibling. Inheritance on the strength of one shared
# field is exactly the kinship claim that went wrong above.


@dataclass(slots=True)
class Procedure:
    """One derivation attached to a statement — a proof, a solution, a
    calculation.

    The procedure extractor reads the group's full text (via the owning
    Statement's ``statement_of``) and extracts the procedure portion into
    ``content``. Step decomposition is a future pass that will write ``:Act``
    nodes; no ``steps`` field here.
    """

    index: int = 0
    content: str | None = None


@dataclass(slots=True)
class Statement:
    """A pedagogical statement — definition, theorem, exercise, etc.

    The PCF groups related content into compound spans; the role typer
    diagnoses each group's composition. ``statement_of`` carries the group's
    member node ids so the statement extractor can read the full group text
    and extract the statement portion into ``content``. ``procedures`` holds
    zero or one Procedure (real or placeholder).

    ``id`` is the id of the group's FIRST member node — the statement's
    document-order position, and what slots it into the persisted chain in
    that member's place. It names a place in the stream; it is not a node.
    REQUIRED, unlike ``ASTNode.id``: a node exists before the stream is
    flattened and is stamped with its id afterwards, but a statement is only
    ever built from an already-stamped node, so an id-less one is a bug and
    raises at construction rather than travelling on to name nothing.
    """

    id: int
    content: str | None = None
    statement_of: list[int] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)


# --- Picture / Segment ------------------------------------------------------


@dataclass(slots=True)
class Picture:
    """An image extracted from a page.

    `index` is the 1-based placeholder id that OCR's ![N]() markers refer to.
    """

    index: int
    image_path: str


@dataclass(slots=True)
class Segment:
    """One page of the document. `index` is 0-based document order.

    TRANSIENT — per-page ingestion scaffolding. Only its pictures outlive the
    seam merger, and only for placeholder resolution at assembly; no Segment
    reaches the graph.
    """

    index: int
    image_path: str
    pictures: list[Picture] = field(default_factory=list)
    content: str | None = None  # markdown, filled by the OCR stage
    nodes: list[ASTNode] = field(
        default_factory=list
    )  # filled by the extractor stage


# --- Helpers (pure functions over the models above) --------------------------


def merge_results_into_segments(
    segments: list[Segment], results: list[tuple[int, Any]], attr: str
) -> list[Segment]:
    """Drain a stage's reducer channel back into the segment backbone.

    Args:
        segments: The ordered segment backbone.
        results: The stage's ``(segment_index, value)`` entries.
        attr: The Segment attribute each value is written to.

    Returns:
        The same segment list, with ``attr`` set on every segment that had a
        result.
    """
    by_index = dict(results)
    for segment in segments:
        if segment.index in by_index:
            setattr(segment, attr, by_index[segment.index])
    return segments


def flatten_segments(segments: list[Segment]) -> list[ASTNode]:
    """Flatten per-page segment nodes into one global ordered node list.

    Args:
        segments: The ordered segment backbone.

    Returns:
        The flat stream, each node stamped with a stable ``id`` and its
        originating ``segment_index``.
    """
    flat: list[ASTNode] = []
    for segment in segments:
        for node in segment.nodes or []:
            node.segment_index = segment.index
            flat.append(node)
    for i, node in enumerate(flat):
        node.id = i
    return flat
