r"""
Group finder — a cursor-walk over the flat structural node stream that lifts out the
pedagogical blocks and the worked procedures that derive them.

The single detection stage of the entity layer. It is a forward walk:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead
    window* of whole nodes up to a soft token budget, and the LLM returns the
    spans inside it, each an inclusive [start, end] range of local positions
    carrying a role: ``entity`` (a block) or ``procedure`` (its derivation).
  * How far the cursor advances is decided *structurally* — no self-report from the
    LLM. A span is only "banked" once a node is seen to follow it, so it can never
    be split by a window cut:
      - bank every span whose end is BEFORE the window's edge (a node follows it →
        bounded) and advance the cursor to just after the last such span;
      - if the ONLY span reaches the window's edge, it may continue past the cut — so
        instead of banking it, GROW the window (double the budget) and re-read from the
        same cursor, repeating until a node follows it (bounded) or the document ends.
    Growing, not rewinding, means "a block bigger than the window" stops being a
    special case: the window just expands until the block is whole, so nothing is
    ever truncated and no size guard is needed. Termination is automatic — growth
    strictly increases and eventually reaches the document end, which banks the final
    span outright. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a pathologically long
    span to the model's context (banked as-is at the cap); that is a resource limit
    at the edge of the system, not part of the core rule.

The banking machinery above is kept verbatim from the per-type finders it replaces — it is
the reliable half and is deliberately not redesigned. What changed is *what* is detected.

Design commitments:
  * DOMAIN-NEUTRAL, GENRE-SPECIFIC. A labeled pedagogical block is a universal of
    textbooks — math, physics, CS and biology all carry definitions, statements of fact,
    worked examples and exercises — so the finder is domain-free. What *kind* of block it
    is (definition / theorem / law / example / mechanism) is NOT decided here: the finder
    emits spans only, and the open ``type`` is induced downstream by the statement
    extractor.
  * STATEMENT AND PROCEDURE ARE SEPARATE SPANS. A theorem and its proof are two adjacent
    spans, not one fused span split later. The ``entity``/``procedure`` distinction is
    closed, binary and structural — textbooks mark derivations explicitly ("Proof.",
    "Solution.") — which is exactly the kind of judgment this walk is good at. It replaces
    the old per-type attributors' semantic ``proof_start`` / ``solution_start`` boundary
    call with a detection.
  * ENTITIES ARE A SPARSE OVERLAY, and ONE PARTITION. Nodes keep their stable ids; a span
    just records the node ids that are its members. Nothing about the node list is mutated
    or renumbered, so the forward walk emits spans already in document order. Unlike the
    three per-type finders this replaces, a node now belongs to at most one span.

``GroupFinderNode`` (bottom of this file) runs the walk and writes the entity overlay to
the ``entities`` channel and the procedure spans to ``procedure_spans``. The group itself —
the run of nodes a block and its derivation jointly occupy — is a scaffold: it exists only
inside this walk and is never persisted.
"""

import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget still
# forms a window (at least one node). When the only span in a window reaches its edge,
# the window grows (doubling) until it is bounded or the document ends — capped so a
# pathological block can't grow past the model's context (banked as-is there).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000

# The two span roles. Closed, binary and structural — NOT the open `type` taxonomy
# (definition / theorem / law / …), which the statement extractor induces downstream.
ENTITY_ROLE = 'entity'
PROCEDURE_ROLE = 'procedure'
SPAN_ROLES = [ENTITY_ROLE, PROCEDURE_ROLE]


def _estimate_tokens(node: models.ASTNode) -> int:
    return len(node.content or '') // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position, its content, and its role
    annotation (the instruction finder marks an exercise lead-in with role "instruction")."""

    position: int
    type: str
    content: str | None = None
    role: str = ''


class Span(BaseModel):
    """One span the LLM found, as an inclusive range of local positions plus its role."""

    start: int = Field(
        description='First local position of the span (inclusive).'
    )
    end: int = Field(description='Last local position of the span (inclusive).')
    role: str = Field(
        description='Either "entity" (a pedagogical block) or "procedure" (a worked derivation).'
    )


class Signature(dspy.Signature):
    r"""
    Find the PEDAGOGICAL BLOCKS in a run of textbook nodes, and the WORKED PROCEDURES that
    derive them, and return each as a span of node positions. Anchor on the node that opens
    a block, gather the run of nodes that belongs to it, stop at its boundary. This is
    domain-neutral — it applies to ANY textbook (math, physics, CS, biology), not just math.

    TWO KINDS OF SPAN. Give every span a `role`:

    role="entity" — a labeled pedagogical BLOCK. This covers every kind of block a textbook
    presents as a unit:
    - a DECLARATIVE STATEMENT: a definition, theorem, proposition, lemma, corollary, axiom,
      or a domain's law / model / rule / principle.
    - a WORKED EXAMPLE from the exposition (labelled "Example ...", a posed question).
    - an EXERCISE from a problem set (labelled by a number, usually with no solution shown).
    Do NOT classify which kind it is — that is decided later. Just mark it "entity".

    role="procedure" — the WORKED DERIVATION that resolves a block: a proof, a solution, a
    derivation, a worked calculation. A procedure is its OWN span, ALWAYS separate from
    the block it derives — never merge a theorem with its proof, or an example with its
    solution, into one span. If a block has TWO derivations ("Proof 1 ...", "Proof 2 ..."),
    emit TWO procedure spans.

    A MARKER IS COMMON BUT NEVER REQUIRED. Many books mark a derivation explicitly ("Proof.",
    "Solution.", "Proof of Theorem 2.4.") — but MANY DO NOT. A worked example very often runs
    straight from the posed task into the working with no marker word at all. The ABSENCE of
    "Solution." IS NOT evidence that there is no procedure. Decide by what the text DOES, not
    by whether it is labelled:
    - POSING / STATING ("Solve $y' = y^2$, $y(0)=A$.", "Show that $f$ is bounded.", a theorem's
      claim) → that is the entity.
    - WORKING ("We know how to solve this equation. First assume ... so ... hence ...",
      integrating, substituting, case-splitting, computing, concluding) → that is a PROCEDURE.
    Cut the span AT THAT TURN — where the text stops posing or asserting and starts deriving —
    and give the working its own procedure span. An example whose solution is "integrated"
    into it still has a procedure: split it at the turn. Only when the block genuinely shows
    NO working at all (a bare exercise for the reader, a definition, an unproved statement)
    is there no procedure span.

    NOT SPANS AT ALL: ordinary narrative prose, section headers, figures, running text
    between blocks. Return nothing for them.

    EXERCISE LEAD-INS ARE BOUNDARIES, NEVER MEMBERS. A grouped-exercise LEAD-IN — a directive
    that introduces a run of exercises and states a shared instruction ("For the following
    exercises, find the domain and range.", "In Exercises 3-8, graph the given relation.") — is
    NOT a block and is NEVER part of any span. Such a node has NO number of its own and is
    marked with role "instruction"; treat it exactly like a section header — a boundary. The
    exercises it governs are SEPARATE blocks that FOLLOW it: begin the first one at the first
    exercise node AFTER the lead-in, never at the lead-in itself, and never extend a preceding
    span forward to absorb it.

    EXTENT (what nodes a span includes):
    - START at the block's OWN label/heading. A block usually opens with a short label that
      is a SEPARATE node from its text — e.g. a node that is just "Example 6.7",
      "Definition 3.1", "Theorem 2.5.8", or "Exercise 12". That label node is the FIRST node
      of the span: ALWAYS include it and begin there, not at the text node after it. (A
      block's own label is NOT the same as a section heading like "Matrix Operations", which
      names a section and is a boundary — never part of a span. When a heading names a
      specific block, it belongs to that block; when it names a section, it does not.)
    - Keep subparts together: a stem with parts (a)(b)(c) or (i)(ii)(iii) is ONE block; a
      repeated base number with letter suffixes (12a, 12b, 12c) is ONE block. Do NOT split
      subparts into separate spans.
    - A procedure span starts at its marker node ("Proof.", "Solution.") when there is one,
      and otherwise at the FIRST node that starts working the block out — and runs to the end
      of the derivation.
    - Stop at the boundary: the next block's label, a procedure marker, the turn from posing
      or stating into working, a section header, an exercise lead-in (role "instruction"), or
      a clear return to ordinary narrative.

    SEPARATE BLOCKS: distinct base numbers are distinct blocks (exercise 12 and exercise 13
    are two spans, never merged). A worked example and a following exercise are two blocks.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a span is the
      inclusive [start, end] range it occupies.
    - Spans must NOT overlap: every node belongs to at most one span.
    - Return the spans in document order.
    - Include a span even if it is unfinished at the last given node — still emit it,
      spanning it out to that last node.
    - If there are no blocks or procedures in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position and a role "
        '(role "instruction" marks an exercise lead-in — a boundary, never part of a span). '
        'Emit spans over these only.'
    )
    spans: list[Span] = dspy.OutputField(
        description='The blocks and procedures found in current_nodes, as position spans with a '
        'role of "entity" or "procedure", in document order. Empty list if none.'
    )


class Module(dspy.Module):
    """Finds pedagogical block and procedure spans in the node stream."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[Span]:
        """Returns the spans found in the given window of nodes."""
        result = await self.finder.acall(current_nodes=current_nodes)
        spans = list(result.spans or [])
        logger.debug(
            'find: %d nodes in, %d spans out (%s) | first node %r',
            len(current_nodes),
            len(spans),
            logs.counts([span.role for span in spans]),
            logs.elide(current_nodes[0].content if current_nodes else ''),
        )
        return spans


def _window_from(nodes: list[models.ASTNode], cursor: int, budget: int) -> int:
    """Return the exclusive end index of a look-ahead window starting at `cursor`:
    whole nodes up to the soft token budget, always at least one node."""
    i, accumulated = cursor, 0
    node_count = len(nodes)
    while i < node_count:
        token_count = _estimate_tokens(nodes[i])
        if i > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        i += 1
    return i


def _clean_spans(spans: list[Span], last_local: int) -> list[Span]:
    """Clamp spans into the window, normalise their role, drop overlaps, and sort.

    A span whose role the model did not give as one of ``SPAN_ROLES`` falls back to
    ``entity`` — the far more common kind, and the one a mislabelled block should join.
    Overlap is resolved greedily in document order (the first span wins), enforcing the
    one partition the schema requires."""
    clamped: list[Span] = []
    for span in spans:
        start = min(max(span.start, 0), last_local)
        end = min(max(span.end, start), last_local)
        role = span.role if span.role in SPAN_ROLES else ENTITY_ROLE
        clamped.append(Span(start=start, end=end, role=role))
    clamped.sort(key=lambda span: (span.start, span.end))

    kept: list[Span] = []
    for span in clamped:
        if kept and span.start <= kept[-1].end:
            continue  # overlaps an already-kept span — one node, one span
        kept.append(span)
    return kept


async def find_groups(
    nodes: list[models.ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> tuple[list[models.Entity], list[list[int]]]:
    """Cursor-walk the node stream and return the block overlay and the procedure spans.

    From the cursor, read a look-ahead window and ask the LLM for the spans in it. Bank
    every span a node is seen to follow (bounded) and advance past them; if the only span
    reaches the window's edge it may continue, so grow the window and re-read from the same
    cursor until a node follows it (bounded) or the document ends. Growing — never
    rewinding — captures a block larger than the window whole rather than truncating it,
    and needs no size guard: growth terminates at the document end (or the ``max_budget``
    context cap, the one place a rare truncation can remain).

    Args:
        nodes: The flat, document-ordered node stream.
        module: The finder module. Created fresh if None.
        budget: Soft token budget for the initial look-ahead window.
        max_budget: Cap past which a growing window is banked as-is.

    Returns:
        An ``(entities, procedure_spans)`` pair, both in document order. Entities carry
        their member node ids and nothing else — attributes come from the statement
        extractor. Each procedure span is a list of member node ids.
    """
    module = module or Module()
    entities: list[models.Entity] = []
    procedure_spans: list[list[int]] = []
    cursor, node_count = 0, len(nodes)

    while cursor < node_count:
        size = budget
        while True:
            end = _window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            last_local = len(window) - 1
            reached_doc_end = end == node_count

            spans = await module.aforward(
                [
                    WindowNode(
                        position=k,
                        type=(node.type.value if node.type else ''),
                        content=node.content,
                        role=(node.role or ''),
                    )
                    for k, node in enumerate(window)
                ]
            )
            clean = _clean_spans(spans, last_local)

            if not clean:
                cursor = end  # only prose in this window — skip it
                break

            # A span is bounded when a node is seen to follow it inside the window.
            bounded = [span for span in clean if span.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the context
                # cap: bank every span as-is and advance past the window.
                if not reached_doc_end:
                    logger.warning(
                        'window hit the %d-token cap at cursor %d; banking %d '
                        'span(s) as-is (a span may be truncated)',
                        max_budget,
                        cursor,
                        len(clean),
                    )
                to_bank, advance = clean, end
            elif bounded:
                # Commit the bounded spans; the cursor lands just after the last one
                # (any trailing prose / an unbanked edge span is re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole span reaches the edge and may continue — grow and re-read.
                logger.debug(
                    'grow: sole span reaches the window edge at cursor %d; '
                    'budget %d -> %d',
                    cursor,
                    size,
                    size * 2,
                )
                size *= 2
                continue

            for span in to_bank:
                ids = [
                    window[k].id
                    for k in range(span.start, span.end + 1)
                    if window[k].id is not None
                ]
                if not ids:
                    continue
                if span.role == PROCEDURE_ROLE:
                    procedure_spans.append(ids)
                else:
                    entities.append(models.Entity(members=ids))
            cursor = advance
            break

    # The two counts are the stage's headline result: blocks found, and how many of them
    # the book shows worked out. A book with blocks but zero procedure spans is the
    # signature of an unmarked-derivation miss (docs/HANDOFF.md, next step 1).
    logger.info(
        'group finder: %d nodes -> %d block(s), %d procedure span(s)',
        node_count,
        len(entities),
        len(procedure_spans),
    )
    return entities, procedure_spans


# --- LangGraph node: emit the found blocks and procedure spans onto their channels ---


class GroupFinderNode:
    """Walks the flat node stream and writes the block overlay and the procedure spans.

    The walk is one sequential unit (a growing look-ahead cursor cannot be sharded), so
    this is a plain graph node rather than the map-reduce dispatch/worker/collect shape
    the parallel stages use."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Walks the node stream and writes the entity overlay and procedure spans."""
        entities, procedure_spans = await find_groups(
            state.get('nodes', []), module=self.module
        )
        return {'entities': entities, 'procedure_spans': procedure_spans}
