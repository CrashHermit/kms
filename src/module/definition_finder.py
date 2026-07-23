r"""
Definition finder — a cursor-walk over the flat structural node stream that lifts out
Definition entities (AutoMathKG's Definition type).

Structurally identical to the Problem finder (a self-contained copy of the same
forward walk); only the prompt and the emitted type differ:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead
    window* of whole nodes up to a soft token budget, and the LLM returns the
    Definitions inside it, each as an inclusive [start, end] span of local positions.
  * How far the cursor advances is decided *structurally* — no self-report from the
    LLM. A definition is only "banked" once a node is seen to follow it, so it can never
    be split by a window cut:
      - bank every group whose end is BEFORE the window's edge (a node follows it →
        bounded) and advance the cursor to just after the last such group;
      - if the ONLY group reaches the window's edge, it may continue past the cut — so
        instead of banking it, GROW the window (double the budget) and re-read from the
        same cursor, repeating until a node follows it (bounded) or the document ends.
    Growing, not rewinding, means "a definition bigger than the window" stops being a
    special case: the window just expands until it is whole, so nothing is ever
    truncated and no size guard is needed. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a
    pathologically long entity to the model's context (banked as-is at the cap).

Entities are a sparse OVERLAY: nodes keep their stable ids; a Definition just records
the node ids that are its members. Nothing about the node list is mutated, so the
forward walk emits Definitions already in document order. Per-attribute detail (number,
field, …) is a later per-attribute pass.

The finder is wired into the pipeline by ``DefinitionFinderNode`` (bottom of this file),
which runs the walk and writes its Definition entities to the ``definition_entities``
channel, in parallel with the Problem and Theorem finders.
"""

import dspy
from pydantic import BaseModel, Field

from .llm import text_lm
from .state import ASTNode, Entity, EntityType, State

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget still
# forms a window (at least one node). When the only definition in a window reaches its
# edge, the window grows (doubling) until it is bounded or the document ends — capped so
# a pathological entity can't grow past the model's context (banked as-is there).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class DefinitionSpan(BaseModel):
    """A Definition the LLM found, as an inclusive span of local positions in the window."""

    start: int = Field(description="First local position of the definition (inclusive).")
    end: int = Field(description="Last local position of the definition (inclusive).")


class Signature(dspy.Signature):
    r"""
    Find the mathematical DEFINITIONS in a run of textbook nodes and return each as a
    span of node positions. Anchor on the node that opens a definition, gather the run of
    nodes that belongs to it, stop at its boundary.

    WHAT IS A DEFINITION:
    A Definition is a marked, self-contained statement that introduces and fixes the
    meaning of a mathematical concept or term — the place the text formally says "we
    define X to be ...". It is usually set off from the surrounding prose, often labelled
    ("Definition", "Definition 2.1") or written as a clearly-marked defining statement.

    Theorems, propositions, lemmas, corollaries, their proofs, worked examples, and
    exercises are NOT definitions — ignore them. Ordinary narrative prose that merely
    uses or mentions a term without formally fixing its meaning, figures, and section
    headers are not definitions either.

    EXTENT (what nodes a definition's span includes):
    - START at the definition's OWN label/heading. A definition usually opens with a
      short label that is a SEPARATE node from its statement — e.g. a node that is just
      "Definition 2.1" or "Definition". That label node is the FIRST node of the
      definition: ALWAYS include it and begin the span there, not at the statement node
      after it. (A definition's own label is NOT the same as a section heading like
      "Vector Spaces", which names a section and is a boundary — never part of a span.)
    - Its defining statement, together with any nodes that complete it — a defining
      display-math block, or a short clarifying clause / condition list that is part of
      the same definition.
    - A definition has no proof and no solution. Stop at the boundary: the next label
      (definition/theorem/example/exercise), a section header, or a clear return to
      ordinary narrative.

    SEPARATE DEFINITIONS: distinct definitions are distinct spans (Definition 2.1 and
    Definition 2.2 are two spans, never merged), even when they sit back to back.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a span is the
      inclusive [start, end] range it occupies.
    - Return the definitions in document order.
    - Include a definition even if it is unfinished at the last given node — still emit
      it, spanning it out to that last node.
    - If there are no definitions in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position. Emit spans over these only."
    )
    definitions: list[DefinitionSpan] = dspy.OutputField(
        description="The definitions found in current_nodes, as position spans, in document order. Empty list if none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[DefinitionSpan]:
        result = await self.finder.acall(current_nodes=current_nodes)
        return list(result.definitions or [])


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


async def find_definitions(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[Entity]:
    """Cursor-walk the node stream and return Definition entities (sparse overlay).

    From the cursor, read a look-ahead window and ask the LLM for the definitions in it.
    Bank every definition a node is seen to follow (bounded) and advance past them; if
    the only definition reaches the window's edge it may continue, so grow the window and
    re-read from the same cursor until a node follows it (bounded) or the document ends.
    """
    module = module or Module()
    definitions: list[Entity] = []
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
            clean: list[DefinitionSpan] = []
            for s in spans:
                start = min(max(s.start, 0), last_local)
                stop = min(max(s.end, start), last_local)
                clean.append(DefinitionSpan(start=start, end=stop))
            clean.sort(key=lambda s: s.start)

            if not clean:
                cursor = end  # only prose in this window — skip it
                break

            # A definition is bounded when a node is seen to follow it inside the window.
            bounded = [s for s in clean if s.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the context
                # cap: bank every definition as-is and advance past the window.
                to_bank, advance = clean, end
            elif bounded:
                # Commit the bounded definitions; the cursor lands just after the last one
                # (any trailing prose / an unbanked edge definition is re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole definition reaches the edge and may continue — grow and re-read.
                size *= 2
                continue

            for s in to_bank:
                ids = [window[k].id for k in range(s.start, s.end + 1) if window[k].id is not None]
                if ids:
                    definitions.append(Entity(type=EntityType.DEFINITION, members=ids))
            cursor = advance
            break

    return definitions


# --- LangGraph node: emit the found Definitions onto their channel ---


class DefinitionFinderNode:
    """Walks the flat node stream and writes its Definition entities to the
    ``definition_entities`` channel.

    The walk is one sequential unit (a growing look-ahead cursor cannot be sharded), so
    this is a plain graph node rather than the map-reduce dispatch/worker/collect shape
    the parallel stages use."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        definitions = await find_definitions(state.get("nodes", []), module=self.module)
        return {"definition_entities": definitions}
