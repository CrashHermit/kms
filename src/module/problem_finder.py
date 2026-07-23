r"""
Problem finder — a cursor-walk over the flat structural node stream that lifts out
Problem entities (worked examples AND exercises; AutoMathKG's Problem type).

This is the first of the per-type finders (Problem / Definition / Theorem). It is a
single forward walk:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead
    window* of whole nodes up to a soft token budget, and the LLM returns the
    Problems inside it, each as an inclusive [start, end] span of local positions.
  * How far the cursor advances is decided *structurally* — no self-report from the
    LLM. A problem is only "banked" once a node is seen to follow it, so it can never
    be split by a window cut:
      - bank every group whose end is BEFORE the window's edge (a node follows it →
        bounded) and advance the cursor to just after the last such group;
      - if the ONLY group reaches the window's edge, it may continue past the cut — so
        instead of banking it, GROW the window (double the budget) and re-read from the
        same cursor, repeating until a node follows it (bounded) or the document ends.
    Growing, not rewinding, means "a problem bigger than the window" stops being a
    special case: the window just expands until the problem is whole, so nothing is
    ever truncated and no size guard is needed. Termination is automatic — growth
    strictly increases and eventually reaches the document end, which banks the final
    problem outright. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a pathologically long
    problem to the model's context (banked as-is at the cap); that is a resource limit
    at the edge of the system, not part of the core rule.

Design commitments:
  * A Problem is any posed mathematical task — worked example or exercise — regardless
    of whether a solution is shown. The finder is *solution-agnostic*: it captures the
    problem's whole extent (statement, subparts, and a solution if one happens to be
    present) but does NOT label statement-vs-solution. Roles, number, instruction, and
    field are later per-attribute passes.
  * Entities are a sparse OVERLAY: nodes keep their stable ids; a Problem just records
    the node ids that are its members. Nothing about the node list is mutated or
    renumbered, so the forward walk emits Problems already in document order.

The finder is wired into the pipeline by ``ProblemFinderNode`` (bottom of this file),
which runs the walk over the flat node stream and writes its Problem entities to the
``problem_entities`` channel. It runs in parallel with the Definition and Theorem finders
(each a self-contained copy of this same cursor-walk, writing its own channel). The three
overlays are independent and may reference the same node from more than one entity — that
is fine, since members are node-id pointers, not copies.
"""

import dspy
from pydantic import BaseModel, Field

from .llm import text_lm
from .state import ASTNode, Entity, EntityType, State

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget still
# forms a window (at least one node). When the only problem in a window reaches its edge,
# the window grows (doubling) until it is bounded or the document ends — capped so a
# pathological problem can't grow past the model's context (banked as-is there).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position, its content, and its role
    annotation (the splitter marks an exercise lead-in with role "instruction")."""

    position: int
    type: str
    content: str | None = None
    role: str = ""


class ProblemSpan(BaseModel):
    """A Problem the LLM found, as an inclusive span of local positions in the window."""

    start: int = Field(description="First local position of the problem (inclusive).")
    end: int = Field(description="Last local position of the problem (inclusive).")


class Signature(dspy.Signature):
    r"""
    Find the mathematical PROBLEMS in a run of textbook nodes and return each as a
    span of node positions. Anchor on the node that opens a problem, gather the run of
    nodes that belongs to it, stop at its boundary.

    WHAT IS A PROBLEM:
    A Problem is any posed mathematical task — this covers BOTH:
    - a worked EXAMPLE from the exposition (labelled "Example ...", a posed question,
      usually with a shown solution), and
    - an EXERCISE from a problem set (labelled by a number, usually with NO solution —
      the reader is meant to solve it).
    Treat both the same: they are Problems. Do NOT require a solution to be present — a
    problem whose solution is left to the reader is still a Problem.

    Definitions, theorems, propositions, lemmas, corollaries, and their proofs are NOT
    problems — ignore them. Ordinary narrative prose, figures, and section headers are
    not problems either.

    EXERCISE LEAD-INS ARE BOUNDARIES, NEVER MEMBERS. A grouped-exercise LEAD-IN — a directive
    that introduces a run of exercises and states a shared instruction ("For the following
    exercises, find the domain and range.", "In Exercises 3-8, graph the given relation.") — is
    NOT a problem and is NEVER part of any problem's span. Such a node has NO exercise number of
    its own and is marked with role "instruction"; treat it exactly like a section header — a
    boundary. The exercises it governs are SEPARATE problems that FOLLOW it: begin the first one
    at the first exercise node AFTER the lead-in, never at the lead-in itself, and never extend a
    preceding problem forward to absorb it. (A later pass attaches the lead-in's shared
    instruction to those problems; the finder's only job here is to not swallow the lead-in.)

    EXTENT (what nodes a problem's span includes):
    - START at the problem's OWN label/heading. A problem usually opens with a short
      label that is a SEPARATE node from its question text — e.g. a node that is just
      "Example 6.7", "6.3 Check Your Understanding", or "Exercise 12". That label node
      is the FIRST node of the problem: ALWAYS include it and begin the span there, not
      at the question node after it. (A problem's own label is NOT the same as a section
      heading like "Matrix Operations", which names a section and is a boundary — never
      part of a span. When a heading names a specific problem, it belongs to that
      problem; when it names a section, it does not.)
    - Its statement/question, with subparts kept together: a stem with parts (a)(b)(c)
      or (i)(ii)(iii) is ONE problem; a repeated base number with letter suffixes
      (12a, 12b, 12c) is ONE problem. Do NOT split subparts into separate problems.
    - Its solution/answer nodes IF shown (prose, display math, steps) — include them in
      the same span. Do not include them if none is shown.
    - Stop at the boundary: the next problem's label, a section header, an exercise lead-in
      (role "instruction"), a definition/theorem, or a clear return to ordinary narrative.

    SEPARATE PROBLEMS: distinct base numbers are distinct problems (exercise 12 and
    exercise 13 are two spans, never merged). A worked example and a following exercise
    are two problems.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a span is the
      inclusive [start, end] range it occupies.
    - Return the problems in document order.
    - Include a problem even if it is unfinished at the last given node — still emit it,
      spanning it out to that last node.
    - If there are no problems in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position and a role "
        '(role "instruction" marks an exercise lead-in — a boundary, never part of a span). '
        "Emit spans over these only."
    )
    problems: list[ProblemSpan] = dspy.OutputField(
        description="The problems found in current_nodes, as position spans, in document order. Empty list if none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[ProblemSpan]:
        result = await self.finder.acall(current_nodes=current_nodes)
        return list(result.problems or [])


def _window_from(nodes: list[ASTNode], cursor: int, budget: int) -> int:
    """Return the exclusive end index of a look-ahead window starting at `cursor`:
    whole nodes up to the soft token budget, always at least one node."""
    i, tokens = cursor, 0
    n = len(nodes)
    while i < n:
        t = _est_tokens(nodes[i])
        if i > cursor and tokens + t > budget:
            break
        tokens += t
        i += 1
    return i


async def find_problems(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[Entity]:
    """Cursor-walk the node stream and return Problem entities (sparse overlay).

    From the cursor, read a look-ahead window and ask the LLM for the problems in it.
    Bank every problem a node is seen to follow (bounded) and advance past them; if the
    only problem reaches the window's edge it may continue, so grow the window and
    re-read from the same cursor until a node follows it (bounded) or the document ends.
    Growing — never rewinding — captures a problem larger than the window whole rather
    than truncating it, and needs no size guard: growth terminates at the document end
    (or the ``max_budget`` context cap, the one place a rare truncation can remain).
    """
    module = module or Module()
    problems: list[Entity] = []
    cursor, n = 0, len(nodes)

    while cursor < n:
        size = budget
        while True:
            end = _window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            last_local = len(window) - 1
            reached_doc_end = end == n

            spans = await module.aforward(
                [
                    WindowNode(
                        position=k,
                        type=(node.type.value if node.type else ""),
                        content=node.content,
                        role=(node.role or ""),
                    )
                    for k, node in enumerate(window)
                ]
            )
            # Clamp to range, drop empties, keep document order.
            clean: list[ProblemSpan] = []
            for s in spans:
                start = min(max(s.start, 0), last_local)
                stop = min(max(s.end, start), last_local)
                clean.append(ProblemSpan(start=start, end=stop))
            clean.sort(key=lambda s: s.start)

            if not clean:
                cursor = end  # only prose in this window — skip it
                break

            # A problem is bounded when a node is seen to follow it inside the window.
            bounded = [s for s in clean if s.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the context
                # cap: bank every problem as-is and advance past the window.
                to_bank, advance = clean, end
            elif bounded:
                # Commit the bounded problems; the cursor lands just after the last one
                # (any trailing prose / an unbanked edge problem is re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole problem reaches the edge and may continue — grow and re-read.
                size *= 2
                continue

            for s in to_bank:
                ids = [window[k].id for k in range(s.start, s.end + 1) if window[k].id is not None]
                if ids:
                    problems.append(Entity(type=EntityType.PROBLEM, members=ids))
            cursor = advance
            break

    return problems


# --- LangGraph node: emit the found Problems onto their channel ---


class ProblemFinderNode:
    """Walks the flat node stream and writes its Problem entities to the
    ``problem_entities`` channel.

    The walk is one sequential unit (a growing look-ahead cursor cannot be sharded), so
    this is a plain graph node rather than the map-reduce dispatch/worker/collect shape
    the parallel stages use."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        problems = await find_problems(state.get("nodes", []), module=self.module)
        return {"problem_entities": problems}
