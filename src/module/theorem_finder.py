r"""
Theorem finder — a cursor-walk over the flat structural node stream that lifts out
Theorem entities (AutoMathKG's Theorem type, which SUBSUMES propositions, corollaries,
and lemmas).

Structurally identical to the Problem finder (a self-contained copy of the same
forward walk); only the prompt and the emitted type differ:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead
    window* of whole nodes up to a soft token budget, and the LLM returns the
    Theorems inside it, each as an inclusive [start, end] span of local positions.
  * How far the cursor advances is decided *structurally* — no self-report from the
    LLM. A theorem is only "banked" once a node is seen to follow it, so it can never
    be split by a window cut:
      - bank every group whose end is BEFORE the window's edge (a node follows it →
        bounded) and advance the cursor to just after the last such group;
      - if the ONLY group reaches the window's edge, it may continue past the cut — so
        instead of banking it, GROW the window (double the budget) and re-read from the
        same cursor, repeating until a node follows it (bounded) or the document ends.
    Growing, not rewinding, means "a theorem+proof bigger than the window" stops being a
    special case: the window just expands until it is whole, so nothing is ever
    truncated and no size guard is needed. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a
    pathologically long entity to the model's context (banked as-is at the cap).

Entities are a sparse OVERLAY: nodes keep their stable ids; a Theorem just records the
node ids that are its members (its statement plus its proof nodes). Nothing about the
node list is mutated, so the forward walk emits Theorems already in document order.
Statement-vs-proof roles are a later per-attribute pass — the finder captures extent, not
roles.

The finder is wired into the pipeline by ``TheoremFinderNode`` (bottom of this file),
which runs the walk and writes its Theorem entities to the ``theorem_entities`` channel,
in parallel with the Problem and Definition finders.
"""

import dspy
from pydantic import BaseModel, Field

from .llm import text_lm
from .state import ASTNode, Entity, EntityType, State

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget still
# forms a window (at least one node). When the only theorem in a window reaches its edge,
# the window grows (doubling) until it is bounded or the document ends — capped so a
# pathological entity can't grow past the model's context (banked as-is there). A theorem
# plus a long proof is the common reason the window has to grow.
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class TheoremSpan(BaseModel):
    """A Theorem the LLM found, as an inclusive span of local positions in the window."""

    start: int = Field(description="First local position of the theorem (inclusive).")
    end: int = Field(description="Last local position of the theorem (inclusive).")


class Signature(dspy.Signature):
    r"""
    Find the mathematical THEOREMS in a run of textbook nodes and return each as a span
    of node positions. Anchor on the node that opens a theorem, gather the run of nodes
    that belongs to it (its statement AND its proof if one is shown), stop at its
    boundary.

    WHAT IS A THEOREM:
    A Theorem is a marked, self-contained statement asserting a mathematical claim
    presented as established or provable. This ALSO covers PROPOSITIONS, COROLLARIES, and
    LEMMAS — treat all of them as theorems. A theorem is usually labelled ("Theorem",
    "Theorem 3.2", "Lemma 1", "Corollary 4.5") and is often followed by a proof.

    Definitions, worked examples, exercises, and ordinary narrative prose are NOT
    theorems — ignore them. A proof that follows a theorem statement is NOT a separate
    thing: it is PART of that theorem (see EXTENT).

    EXTENT (what nodes a theorem's span includes):
    - START at the theorem's OWN label/heading. A theorem usually opens with a short
      label that is a SEPARATE node from its claim — e.g. a node that is just
      "Theorem 3.2" or "Lemma 1". That label node is the FIRST node of the theorem:
      ALWAYS include it and begin the span there, not at the claim node after it. (A
      theorem's own label is NOT a section heading like "Convergence", which names a
      section and is a boundary — never part of a span.)
    - Its claim/statement, THEN its proof if one is shown: the "Proof." node and every
      node of the proof after it (prose, display math, steps), through to the end of the
      proof (often marked "□"/"QED" or a clear return to narrative). Keep the whole proof
      in the same span. A theorem may have no proof shown — then the span is just the
      statement.
    - Stop at the boundary: the next label (theorem/definition/example/exercise), a
      section header, the end of the proof, or a clear return to ordinary narrative.

    SEPARATE THEOREMS: distinct theorems are distinct spans. A theorem and a following
    corollary are TWO theorems (two spans), never merged, even though both are
    theorem-type.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a span is the
      inclusive [start, end] range it occupies.
    - Return the theorems in document order.
    - Include a theorem even if it is unfinished at the last given node — still emit it,
      spanning it out to that last node.
    - If there are no theorems in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position. Emit spans over these only."
    )
    theorems: list[TheoremSpan] = dspy.OutputField(
        description="The theorems found in current_nodes, as position spans, in document order. Empty list if none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[TheoremSpan]:
        result = await self.finder.acall(current_nodes=current_nodes)
        return list(result.theorems or [])


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


async def find_theorems(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[Entity]:
    """Cursor-walk the node stream and return Theorem entities (sparse overlay).

    From the cursor, read a look-ahead window and ask the LLM for the theorems in it.
    Bank every theorem a node is seen to follow (bounded) and advance past them; if the
    only theorem reaches the window's edge it may continue (a long proof often does), so
    grow the window and re-read from the same cursor until a node follows it (bounded) or
    the document ends.
    """
    module = module or Module()
    theorems: list[Entity] = []
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
                    )
                    for k, node in enumerate(window)
                ]
            )
            # Clamp to range, drop empties, keep document order.
            clean: list[TheoremSpan] = []
            for s in spans:
                start = min(max(s.start, 0), last_local)
                stop = min(max(s.end, start), last_local)
                clean.append(TheoremSpan(start=start, end=stop))
            clean.sort(key=lambda s: s.start)

            if not clean:
                cursor = end  # only prose in this window — skip it
                break

            # A theorem is bounded when a node is seen to follow it inside the window.
            bounded = [s for s in clean if s.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the context
                # cap: bank every theorem as-is and advance past the window.
                to_bank, advance = clean, end
            elif bounded:
                # Commit the bounded theorems; the cursor lands just after the last one
                # (any trailing prose / an unbanked edge theorem is re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole theorem reaches the edge and may continue — grow and re-read.
                size *= 2
                continue

            for s in to_bank:
                ids = [window[k].id for k in range(s.start, s.end + 1) if window[k].id is not None]
                if ids:
                    theorems.append(Entity(type=EntityType.THEOREM, members=ids))
            cursor = advance
            break

    return theorems


# --- LangGraph node: emit the found Theorems onto their channel ---


class TheoremFinderNode:
    """Walks the flat node stream and writes its Theorem entities to the
    ``theorem_entities`` channel.

    The walk is one sequential unit (a growing look-ahead cursor cannot be sharded), so
    this is a plain graph node rather than the map-reduce dispatch/worker/collect shape
    the parallel stages use."""

    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        theorems = await find_theorems(state.get("nodes", []), module=self.module)
        return {"theorem_entities": theorems}
