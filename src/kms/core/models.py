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
    """One node in the flat document stream.

    The structural kind is a plain string (``paragraph``, ``math``,
    ``header``, …) — no subclass hierarchy.  The extractor assigns it
    and the graph tier reads it back via ``kind``.
    """

    type: str | None = None
    content: str | None = None
    id: int | None = None
    segment_index: int | None = None


# --- Atomic facts ------------------------------------------------------------


@dataclass(slots=True)
class AtomicFact:
    """One atomic fact extracted from the node stream.

    A short, self-contained snippet conveying exactly one piece of
    information, drawn from one or more provenance nodes. Minimal by design:
    no kind, no source — classification is a downstream pass's job, and
    provenance is recoverable by resolving ``node_ids`` into the stream.
    """

    text: str
    node_ids: list[int] = field(default_factory=list)


# --- Semantic overlay -------------------------------------------------------
#
# The overlay sits BESIDE the node stream, never in it. A node is one verbatim
# block of the page; a Statement is a HUB identifying the group of blocks a
# pedagogical unit occupies — it carries no text, its members do. Deliberately
# NOT an ASTNode: while it was one, a Statement could be — and was — assigned
# into the stream in its first member's place, which made every consumer of
# that stream read the group's text twice, and the persister write it twice.
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
class Instruction:
    """A shared lead-in and the exercises it governs.

    A hub, like ``Statement`` and ``Procedure``: it identifies the nodes a
    directive applies to and carries no member text. It differs in where it
    comes from — not a PCF span, but the lead-in node the instruction finder
    tagged, which the distributor then removes from the stream once the
    governance is recorded here.

    A hub rather than text copied onto each governed node, because copying
    put SYNTHESIZED text into the provenance layer: ``directive`` is the
    model's normalised imperative ("simplify"), which is not what the page
    says ("In the following exercises, simplify."), and a ``:Node`` is
    defined as one verbatim block of the page. The page's own sentence lives
    here in ``text``, once, as printed.

    ``node_id`` is the lead-in's own id in the flattened stream, frozen
    before removal. It is the hub's identity (see
    ``graph.instructions.instruction_uuid``) and its document position.
    """

    node_id: int
    text: str
    directive: str | None = None
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


# --- Triplets -------------------------------------------------------------


@dataclass(slots=True)
class Triplet:
    """One (subject, predicate, object) relation extracted from an atomic fact.

    Subject and object are verbatim substrings of the source fact text — no
    normalization, no abstraction.  Canonicalization into entities is a
    downstream pass's job.

    ``fact_index`` is the source fact's position in the document-order fact
    list, set by the entry point as it iterates.
    """

    subject: str
    predicate: str
    object: str
    fact_index: int = -1


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
