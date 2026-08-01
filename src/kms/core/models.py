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


# --- Equation & variable binding ---------------------------------------------


@dataclass(slots=True)
class Equation:
    """One equation extracted from a content node.

    Carries the LaTeX source, an optional identity (e.g. "heat equation")
    resolved against the existing graph, and an optional domain label.
    """

    id: int | None = None
    latex: str | None = None
    name: str | None = None
    domain: str | None = None


@dataclass(slots=True)
class Variable:
    """One stand-in bound to a meaning at a specific position in the text.

    Domain-agnostic: a symbol in a mathematical expression, an element in a
    chemical equation, a labelled component in a circuit diagram, a parameter
    in a function signature, a defined term in a legal document — anything
    where a compact notation stands for a fuller meaning.

    Two kinds of binding, both carried here:

    * **Definitional** — the symbol is bound to a MEANING: "let $\\alpha$ be
      the learning rate", "$b$, the cost of the blouse". ``meaning`` holds it
      and ``value`` is None.
    * **Substitutional** — the symbol is bound to a VALUE, for this passage
      only: "evaluate $x^2 + 5x - 8$ when $x = 6$". ``value`` holds the ``6``.

    ``value`` exists because the substitution IS the exercise: without it two
    exercises over the same expression are indistinguishable in the graph, and
    nothing downstream can pose the question, check an answer, or build a
    review card from the node. It is a string, not a number — a binding is as
    often ``-3``, ``2\\pi``, or ``n+1`` as it is an integer.

    When ``equation_index`` is set, the variable belongs to an equation
    extracted from the same node — the ``:HAS_VARIABLE`` edge will point
    from that ``:Equation`` instead of from the ``:Node``.
    """

    symbol: str
    meaning: str
    kind: str
    equation_index: int | None = None
    value: str | None = None


# --- Semantic overlay -------------------------------------------------------
#
# The overlay sits BESIDE the node stream, never in it. A node is one verbatim
# block of the page; a Statement is a HUB identifying the group of blocks a
# pedagogical unit occupies — it carries no text, its members do. Deliberately
# NOT an ASTNode: while it was one, a Statement could be — and was — assigned
# into the stream in its first member's place, which made every consumer of
# that stream read the group's text twice (the duplicated blocks in the
# assembled output).
#
# Statement and Procedure share no base class, and they share no common
# fields: a Procedure is a Statement's CHILD rather than its sibling, and
# inheritance would be a kinship claim of exactly the kind that went wrong
# above.


@dataclass(slots=True)
class Procedure:
    """One derivation — a proof, a solution, a calculation.

    A hub over the member nodes of its portion; it carries no text of its own.
    ``block`` is the FULL PCF span, frozen at creation — the hub's identity
    (see ``graph.procedures.procedure_uuid``), its association key (a
    both-block's statement shares it), and its document position
    (``block[0]``). ``members`` starts as the whole block and is narrowed to
    the derivation portion by the procedure partitioner. ``index``
    distinguishes multiple derivations within one block. Step decomposition
    is a future pass that will write ``:Act`` nodes; no ``steps`` field here.
    """

    block: list[int]
    index: int = 0
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    """A pedagogical statement — what a block states.

    A HUB, not a text container: the PCF groups related content into compound
    spans and the role typer diagnoses each group's composition, but the
    statement itself only identifies nodes — the raw text lives on those
    nodes. ``block`` is the FULL PCF span, frozen at creation: the hub's
    identity (see ``graph.statements.statement_uuid``), its association key
    (a both-block's procedure shares it), and its document position
    (``block[0]``). ``members`` starts as the whole block and is narrowed to
    the statement portion by the statement partitioner. ``block`` is REQUIRED
    — a hub is only ever built from an already-found span, so a block-less
    one is a bug and raises at construction.
    """

    block: list[int]
    members: list[int] = field(default_factory=list)


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
