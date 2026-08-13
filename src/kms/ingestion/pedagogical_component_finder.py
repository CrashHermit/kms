import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import content, logs, models, recording, state, walker

logger = logging.getLogger(__name__)
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowNode(BaseModel):
    position: int
    type: str
    content: str | None = None
    image_path: str | None = None


class Span(BaseModel):
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

    NOT SPANS AT ALL: ordinary narrative prose, section headers, running text
    between blocks. Return nothing for them.


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
    - Emit spans over the given nodes ONLY, using their `[position]` labels; a
      span is the inclusive [start, end] range it occupies.
    - Return the spans in document order. Each node belongs to at most one
      span.
    - Include a span even if it is unfinished at the last given node — still
      emit it, spanning it out to that last node.
    - If there are no units in the window, return an empty list.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The look-ahead window's nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself. Emit spans '
            'over the `[position]` values only.'
        )
    )
    spans: list[Span] = dspy.OutputField(
        description='Every pedagogical unit found in current_nodes, as position spans, in '
        'document order — declarative statements, worked examples, exercises, '
        'and prescribed procedures alike. Boundaries only — do NOT classify '
        'them. Empty list if none.'
    )


class PedagogicalComponentFinder(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.finder = dspy.Predict(Signature)
        self.finder.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content="**Exercise 1.2.1:** Sketch the slope field for $y' = e^{x-y}$.",
                        ),
                        WindowNode(
                            position=1,
                            type='paragraph',
                            content="**Exercise 1.2.2:** Sketch the slope field for $y' = x^2$.",
                        ),
                        WindowNode(
                            position=2,
                            type='paragraph',
                            content="**Exercise 1.2.3:** Sketch the slope field for $y' = y^2$.",
                        ),
                    ]
                ),
                spans=[
                    Span(start=0, end=0),
                    Span(start=1, end=1),
                    Span(start=2, end=2),
                ],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='**Exercise 1.2.7:** Let $\\{x_n\\}$ be a sequence.',
                        ),
                        WindowNode(
                            position=1,
                            type='list',
                            content='a) Show that $\\lim x_n = 0$ iff $\\lim |x_n| = 0$.',
                        ),
                        WindowNode(
                            position=2,
                            type='list',
                            content='b) Find an example where $\\{|x_n|\\}$ converges and $\\{x_n\\}$ diverges.',
                        ),
                    ]
                ),
                spans=[Span(start=0, end=2)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='**Theorem 2.1.10.** Every bounded monotone sequence converges.',
                        ),
                        WindowNode(
                            position=1,
                            type='paragraph',
                            content='Proof. Assume without loss of generality that the sequence is increasing.',
                        ),
                    ]
                ),
                spans=[Span(start=0, end=1)],
            ).with_inputs('current_nodes'),
        ]
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, current_nodes: list[WindowNode]) -> list[Span]:
        result = await self.finder.acall(
            current_nodes=content.labeled_content_parts(current_nodes)
        )
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
        return asyncio.run(self.aforward(current_nodes))


def _normalize_spans(spans: list[Span], last_local: int) -> list[Span]:
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
                        image_path=node.image_path,
                    )
                    for position, node in enumerate(window)
                ]
            )
            clean = _normalize_spans(spans, last_local)

            if not clean:
                cursor = end
                break
            bounded = [span for span in clean if span.end < last_local]

            if reached_doc_end or size >= max_budget:
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
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
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


class PedagogicalComponentFinderNode:
    def __init__(self, module: PedagogicalComponentFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        nodes = state.get('nodes', [])
        excluded = {
            member
            for instruction in state.get('instructions', [])
            for member in instruction.members
        }
        eligible = [node for node in nodes if node.id not in excluded]
        spans = await find_spans(eligible, module=self.module)
        return {'spans': spans}
