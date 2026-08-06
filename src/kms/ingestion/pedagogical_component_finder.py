r"""
Pedagogical component finder — a cursor-walk over the flat structural node
stream that cuts it into the spans the pedagogical units occupy.

The BOUNDARY stage of the entity layer, and only that: it says where each unit
starts and stops, never what any of them is. The pass downstream answers that
— ``role_typer`` (block or derivation?). Splitting the two keeps this walk on
the job it is reliable at — structural boundary detection — instead of fusing
it with a softer classification call.

It is a forward walk:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead
    window* of whole nodes up to a soft token budget, and the LLM returns the
    spans inside it, each an inclusive [start, end] range of local positions.
  * How far the cursor advances is decided *structurally* — no self-report from
    the LLM. A span is only "banked" once a node is seen to follow it, so it can
    never be split by a window cut:
      - bank every span whose end is BEFORE the window's edge (a node follows it
        → bounded) and advance the cursor to just after the last such span;
      - if the ONLY span reaches the window's edge, it may continue past the cut
        — so instead of banking it, GROW the window (double the budget) and
        re-read from the same cursor, repeating until a node follows it
        (bounded) or the document ends. Growing, not rewinding, means "a block
        bigger than the window" stops being a special case: the window just
        expands until the block is whole, so nothing is ever truncated and no
        size guard is needed. Termination is automatic — growth strictly
        increases and eventually reaches the document end, which banks the final
        span outright. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a pathologically
        long span to the model's context (banked as-is at the cap); that is a
        resource limit at the edge of the system, not part of the core rule.

The banking machinery above is kept verbatim from the per-type finders it
replaces — it is the reliable half and is deliberately not redesigned. What
changed is *what* is detected.

Design commitments:
  * DOMAIN-NEUTRAL, GENRE-SPECIFIC. A labeled pedagogical block is a universal
    of textbooks — math, physics, CS and biology all carry definitions,
    statements of fact, worked examples and exercises — so the finder is
    domain-free.
  * BOUNDARIES ONLY, NO CLASSIFICATION. The walk emits untyped spans. Whether a
    span is a block (definition, theorem, example, exercise, law, …) or a
    derivation (proof, solution, calculation) is ``role_typer``'s question, and
    which kind of block it is is a later pass's. The finder clusters WHOLE
    units — including each unit's own working.
  * A BLOCK OWNS ITS WORKING. A theorem and its proof, an example and its
    solution, are ONE span: the cut between a statement and its derivation is
    deliberately NOT made here. The role typer decides whether a block carries
    a statement, a procedure, or both, and the member partitioners find the
    line between the portions.
  * SPANS ARE A SPARSE OVERLAY. Nodes keep their stable ids; a span just records
    the node ids that are its members. Nothing about the node list is mutated or
    renumbered. Spans may overlap — a long paragraph can straddle two units.

``PedagogicalComponentFinderNode`` (bottom of this file) runs the walk and
writes the untyped spans to the ``spans`` channel, which ``role_typer`` then
classifies into the statement and procedure hubs.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import logs, models, recording, state, walker

logger = logging.getLogger(__name__)

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget
# still forms a window (at least one node). When the only span in a window
# reaches its edge, the window grows (doubling) until it is bounded or the
# document ends — capped so a pathological block can't grow past the model's
# context (banked as-is there).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: position, type, content."""

    position: int
    type: str
    content: str | None = None


class Span(BaseModel):
    """One span the LLM found, as an inclusive range of local positions.

    Untyped by design: WHAT the span is — a block or a derivation — is decided
    by the passes downstream (``role_typer``). This stage only says where one
    unit stops and the next begins.
    """

    start: int = Field(
        description='First local position of the span (inclusive).'
    )
    end: int = Field(description='Last local position of the span (inclusive).')


class Signature(dspy.Signature):
    r"""
    Find the BOUNDARIES of every pedagogical unit in a run of textbook nodes,
    and return each unit as a span of node positions. Anchor on the node that
    opens a unit, gather the run of nodes that belongs to it, stop where the
    next one begins. This is domain-neutral — it applies to ANY textbook (math,
    physics, CS, biology), not just math.

    This task is PURELY STRUCTURAL: say WHERE the units start and stop, never
    WHAT they are. Do not classify, label, or name them — a later pass decides
    what kind of span each is. Your only job is to cut the stream in the right
    places. FIND EVERYTHING: a missed unit is a deleted unit.

    WHAT COUNTS AS A UNIT. Emit a span for each of these:
    - a DECLARATIVE STATEMENT: a definition, theorem, proposition, lemma,
      corollary, axiom, or a domain's law / model / rule / principle.
    - a WORKED EXAMPLE from the exposition (labelled "Example ...", a posed
      question).
    - an EXERCISE from a problem set (labelled by a number, usually with no
      solution shown).
    - a PRESCRIBED PROCEDURE: an ordered run of steps the reader is told to
      carry out — a project's "Steps", a lab protocol, an algorithm given as
      numbered instructions. The whole run is ONE span — its lead-in ("We
      recommend proceeding in the following order:") together with every
      numbered step — never one span per step.

    A BLOCK OWNS ITS WORKING — never split a unit from what resolves it. A
    theorem's proof, an example's solution, a posed problem's worked
    calculation, a computation session and its printed output: all of that
    belongs to the SAME span as the block that posed it. Cut where one UNIT
    ends and the NEXT begins — never between a unit and its own derivation. A
    derivation that follows a block is part of that block's span, whether or
    not it is marked ("Proof.", "Solution.", "Proof of Theorem 2.4."). A
    derivation that stands ALONE — no block before it in the document, as in an
    answers section — is a unit in its own right: emit it as its own span.

    NEVER SKIP A LABELLED UNIT. Every node that opens with its own label —
    "Definition 2.5.1", "Theorem 3.4", "Example 6.7", "SAGE Example 2.5.4.",
    "Lemma 1.2", or a bare leading number ("12.", "2.1.12") that numbers a
    problem in a problem set — BEGINS a span, without exception. This holds
    even when the unit is a single node with nothing worked out after it: a
    bare definition that is simply stated, a theorem quoted without proof, an
    exercise with no solution. Such a unit is
    ONE span of one node. Do not pass over a labelled unit merely because there
    is no working attached to it — a missing span here deletes that block from
    the document entirely.

    A NUMBER IS NOT AUTOMATICALLY A LABEL. The steps of a prescribed procedure
    are numbered too ("0.", "1.", "2."), but those numbers ORDER one unit
    rather than NAME several: they run consecutively under a single lead-in and
    each reads as an act to perform, not as a problem to solve. Keep such a run
    in ONE span. Ask what the number does — does it name a problem the book can
    refer back to, or sequence a step inside something already named?

    NOT SPANS AT ALL: ordinary narrative prose, section headers, figures,
    running text between blocks. Return nothing for them.


    EXTENT (what nodes a span includes):
    - START at the block's OWN label/heading. A block usually opens with a short
      label that is a SEPARATE node from its text — e.g. a node that is just
      "Example 6.7", "Definition 3.1", "Theorem 2.5.8", or "Exercise 12". That
      label node is the FIRST node of the span: ALWAYS include it and begin
      there, not at the text node after it. (A block's own label is NOT the same
      as a section heading like "Matrix Operations", which names a section and
      is a boundary — never part of a span. When a heading names a specific
      block, it belongs to that block; when it names a section, it does not.)
    - Keep subparts together: a stem with parts (a)(b)(c) or (i)(ii)(iii) is ONE
      block; a repeated base number with letter suffixes (12a, 12b, 12c) is ONE
      block. Do NOT split subparts into separate spans.
    - Run a unit through its own working: the span covers the block's label and
      posing, then the derivation that resolves it (its proof, solution, or
      calculation) — all the way to where the next unit or ordinary narrative
      begins. A labelled unit that goes on to work itself out is ONE span.
    - Stop at the boundary: the next unit's label, a section header, or a clear
      return to ordinary narrative.

    SEPARATE UNITS: distinct base numbers are distinct units (exercise 12 and
    exercise 13 are two spans, never merged) — this concerns problems in a
    problem set, not the numbered steps of one procedure, which stay together.
    A worked example and a following exercise are two units.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a
      span is the inclusive [start, end] range it occupies.
    - Return the spans in document order. A node MAY belong to more than one
      span (a long paragraph that straddles two units, a caption shared by a
      figure and the example that follows it).
    - Include a span even if it is unfinished at the last given node — still
      emit it, spanning it out to that last node.
    - If there are no units in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position. "
        'Emit spans over these only.'
    )
    spans: list[Span] = dspy.OutputField(
        description='Every pedagogical unit found in current_nodes, as position spans, in '
        'document order — declarative statements, worked examples, exercises, '
        'and prescribed procedures alike. Boundaries only — do NOT classify '
        'them. Empty list if none.'
    )


class PedagogicalComponentFinder(dspy.Module):
    """Finds the pedagogical units' span boundaries in the node stream.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, current_nodes: list[WindowNode]) -> list[Span]:
        """Judge one window.

        Args:
            current_nodes: The window's nodes, each with a local position.

        Returns:
            The spans found in the window, as local position ranges.
        """
        result = await self.finder.acall(current_nodes=current_nodes)
        if self._recorder:
            self._recorder.record(
                'pedagogical_component_finder',
                {'current_nodes': current_nodes},
                result,
            )
        spans = list(result.spans or [])
        logger.debug(
            'find: %d nodes in, %d span(s) out | first node %r',
            len(current_nodes),
            len(spans),
            logs.elide(current_nodes[0].content if current_nodes else ''),
        )
        return spans

    def forward(self, current_nodes: list[WindowNode]) -> list[Span]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes))


def _normalize_spans(spans: list[Span], last_local: int) -> list[Span]:
    """Clamp spans into the window and sort by start position.

    Args:
        spans: The spans the LLM returned.
        last_local: The window's last local position.

    Returns:
        The clamped spans in start order. Overlaps are preserved — a node may
        belong to more than one span.
    """
    clamped: list[Span] = []
    for span in spans:
        start = min(max(span.start, 0), last_local)
        end = min(max(span.end, start), last_local)
        clamped.append(Span(start=start, end=end))
    clamped.sort(key=lambda span: (span.start, span.end))
    return clamped


async def find_spans(
    nodes: list[models.ASTNode],
    module: PedagogicalComponentFinder,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[list[int]]:
    """Cursor-walk the node stream and return the pedagogical units' spans.

    From the cursor, read a look-ahead window and ask the LLM for the spans in
    it. Bank every span a node is seen to follow (bounded) and advance past
    them; if the only span reaches the window's edge it may continue, so grow
    the window and re-read from the same cursor until a node follows it
    (bounded) or the document ends. Growing — never rewinding — captures a
    block larger than the window whole rather than truncating it, and needs no
    size guard: growth terminates at the document end (or the ``max_budget``
    context cap, the one place a rare truncation can remain).

    Args:
        nodes: The flat, document-ordered node stream.
        module: The finder module.
        budget: Soft token budget for the initial look-ahead window.
        max_budget: Cap past which a growing window is banked as-is.

    Returns:
        The spans in document order, each a list of member node ids. UNTYPED —
        whether a span is a block or the derivation that resolves one is
        decided by ``role_typer``.
    """
    module = module
    spans_out: list[list[int]] = []
    cursor, node_count = 0, len(nodes)

    while cursor < node_count:
        size = budget
        while True:
            end = walker.window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            last_local = len(window) - 1
            reached_doc_end = end == node_count

            spans = await module.aforward(
                [
                    WindowNode(
                        position=position,
                        type=node.type,
                        content=node.content,
                    )
                    for position, node in enumerate(window)
                ]
            )
            clean = _normalize_spans(spans, last_local)

            if not clean:
                # Only prose in this window — skip it.
                cursor = end
                break

            # A span is bounded when a node is seen to follow it inside the
            # window.
            bounded = [span for span in clean if span.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the
                # context cap: bank every span as-is and advance past the
                # window.
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
                # Commit the bounded spans; the cursor lands just after the
                # last one (any trailing prose / an unbanked edge span is
                # re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole span reaches the edge and may continue — grow and
                # re-read.
                logger.debug(
                    'grow: sole span reaches the window edge at cursor %d; budget %d -> %d',
                    cursor,
                    size,
                    size * 2,
                )
                size *= 2
                continue

            for span in to_bank:
                member_ids = [
                    window[position].id
                    for position in range(span.start, span.end + 1)
                    if window[position].id is not None
                ]
                if member_ids:
                    spans_out.append(member_ids)
            cursor = advance
            break

    logger.info(
        'pedagogical component finder: %d nodes -> %d span(s)',
        node_count,
        len(spans_out),
    )
    return spans_out


# --- LangGraph node: emit the found spans onto the `spans` channel ---


class PedagogicalComponentFinderNode:
    """Walks the flat node stream and writes the untyped unit spans.

    The walk is one sequential unit (a growing look-ahead cursor cannot be
    sharded), so this is a plain graph node rather than the map-reduce
    dispatch/worker/collect shape the parallel stages use.

    Args:
        module: The finder module.
    """

    def __init__(self, module: PedagogicalComponentFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Walk the node stream and write the untyped spans.

        Args:
            state: The pipeline state, holding the flat node stream.

        Returns:
            The `spans` channel.
        """
        spans = await find_spans(state.get('nodes', []), module=self.module)
        return {'spans': spans}
