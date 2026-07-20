r"""
Entity grouping — the higher-level assembly pass that turns the flat node stream
into a sparse overlay of typed math entities (AutoMathKG's Definition / Theorem /
Problem), one level above the structural nodes and below the knowledge graph.

Two ways into the `entities` list, split by structure rather than by type:

  * Unbounded multi-node spans — Definitions, Theorems, and worked Examples (typed
    `problem`) — are *gathered* from runs of structural nodes. This is the genuinely-
    hard case (ambiguous boundaries, spans that cross pages), so it uses an LLM
    anchor-and-gather pass. The stream is cut into dumb-greedy windows of whole nodes
    (a block is never split; the budget is soft, so one oversized node still forms a
    window). Each window is judged independently — the parallel fan-out — with read-only
    previous/next context so an entity that spills over a window edge can be flagged. A
    single-threaded reconciler then stitches those cross-window spans.

  * Atomic exercises are already bounded — the extractor emits one cohesive `problem`
    node each — so they need no gathering. They are wrapped 1:1 into Problem entities
    mechanically in the collect step, no LLM involved. (Worked Examples are the OTHER
    kind of Problem: unbounded, so they take the gathered path above.)

v1 does entity *extraction* only: an entity is `{id, type, members}`. Roles
(statement/proof/solution/instruction), numbers, and the graph edges come later.

The LLM never handles node ids. It sees each window's current nodes as a local,
0-based positional list and returns entities as position spans; the worker resolves
those positions back to stable global ids. This mirrors the governor/seam pattern:
the LLM judges, code addresses.
"""

import dspy
from pydantic import BaseModel, Field
from langgraph.types import Send

from .state import State, ASTNode, Entity, EntityType, Member, NodeType
from .llm import text_lm

# Soft token budgets (approximate: ~4 chars/token). The current-window budget caps
# how much a single window judges; prev/next cap the read-only context on each side.
CURRENT_BUDGET = 2000
PREV_BUDGET = 500
NEXT_BUDGET = 500

# The types the LLM gathers from runs of prose nodes: definitions, theorems, and
# worked examples (typed `problem`). Atomic `problem` NODES (exercises) are not
# gathered — they are wrapped 1:1 in collect — and are excluded from gathered spans.
_GATHERED = {EntityType.DEFINITION.value, EntityType.THEOREM.value, EntityType.PROBLEM.value}


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One current-window node as the LLM sees it: a local position and its content."""
    position: int
    type: str
    content: str | None = None


class EntitySpan(BaseModel):
    """An entity the LLM found, as an inclusive span of local positions in the window."""
    type: str = Field(description="Entity type: exactly 'definition', 'theorem', or 'problem'.")
    start: int = Field(description="First current-window position of the entity (inclusive).")
    end: int = Field(description="Last current-window position of the entity (inclusive).")
    continues_before: bool = Field(
        default=False,
        description="True if this entity's opening lies before the window — its start continues a run visible only in previous_context.",
    )
    continues_after: bool = Field(
        default=False,
        description="True if this entity is unfinished at the window end — it keeps gathering into next_context.",
    )


class Signature(dspy.Signature):
    r"""
    Find the mathematical Definitions, Theorems, and worked Examples in a run of
    textbook nodes and return each as a span of node positions. This is
    anchor-and-gather: anchor on a cue, gather the run of nodes that belongs to the
    entity, stop at its boundary.

    ENTITY TYPES (general, not tied to any one textbook's formatting):
    - definition: a marked, self-contained statement that introduces and fixes the
      meaning of a mathematical concept or term, set off from the surrounding prose.
    - theorem: a marked, self-contained statement asserting a mathematical claim
      presented as established or provable, together with its proof if one is shown.
      This ALSO covers propositions, corollaries, and lemmas — all are `theorem`.
    - problem: a worked EXAMPLE — a solved problem shown in the exposition, with a
      problem statement AND a worked-out solution. Emit type `problem` for these.

    ANCHOR + GATHER:
    - Anchor on the node that opens a definition, theorem, or worked example — its cue:
      the labelled heading or the opening statement itself. START the span AT that cue
      node. Do NOT include a narrative sentence that merely leads up to the cue: prose
      like "By applying the definition of continuity, we can state the following
      theorem." is connective text, not part of the entity — exclude it and begin the
      span at the "Theorem" cue that follows.
    - Gather forward from the anchor, including every node that is part of that entity:
      a theorem's statement plus the nodes of its proof; a definition's statement and
      any nodes that complete it; a worked example's statement plus every node of its
      solution (the solution may run across several nodes — prose, display math, steps).
    - Stop at the entity's boundary: the next anchor, a section header, an atomic
      problem node (see below), the end of a proof/solution, or a clear return to
      ordinary narrative that is not part of the statement, proof, or solution.

    DO NOT EMIT / DO NOT SPAN:
    - Nodes already of type `problem` are ATOMIC exercises, handled separately. Never
      include such a node in a span and never emit an entity for one. (Worked examples
      you gather are prose nodes, NOT these pre-typed problem nodes.)
    - Ordinary narrative prose, figures, or anything that is not a definition, theorem,
      or worked example. If the window has none, return an empty list.

    POSITIONS:
    - Emit spans over current_nodes ONLY, using their `position` values. A span is the
      inclusive [start, end] range of positions the entity occupies.
    - previous_context and next_context are READ-ONLY neighbouring text for judging
      boundaries. Never emit positions for them.

    WINDOW EDGES:
    - If an entity's statement/cue is NOT in this window but its continuation is (you
      can see the opening in previous_context), emit the in-window part and set
      continues_before = true.
    - If an entity is still gathering when the window ends (its continuation is in
      next_context), set continues_after = true.
    - Otherwise leave both false.
    """

    previous_context: str | None = dspy.InputField(
        description="Read-only markdown of the nodes just before this window. For boundary judgment only."
    )
    current_nodes: list[WindowNode] = dspy.InputField(
        description="The window's nodes, in order, each with a local position. Emit spans over these only."
    )
    next_context: str | None = dspy.InputField(
        description="Read-only markdown of the nodes just after this window. For boundary judgment only."
    )
    entities: list[EntitySpan] = dspy.OutputField(
        description="Definitions and theorems found in current_nodes, as position spans. Empty list if none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.grouper = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(
        self,
        current_nodes: list[WindowNode],
        previous_context: str | None = None,
        next_context: str | None = None,
    ):
        result = await self.grouper.acall(
            previous_context=previous_context,
            current_nodes=current_nodes,
            next_context=next_context,
        )
        entities = result.entities or []
        return dspy.Prediction(entities=entities)


# --- windowing helpers ---

def _windows(nodes: list[ASTNode], budget: int) -> list[tuple[int, int]]:
    """Dumb-greedy chunk the flat stream into [start, end) ranges of whole nodes.
    A node is never split. The budget is soft: a single node larger than the budget
    still forms its own window (at least one node per window)."""
    windows: list[tuple[int, int]] = []
    i, n = 0, len(nodes)
    while i < n:
        start, tokens = i, 0
        while i < n:
            t = _est_tokens(nodes[i])
            if tokens and tokens + t > budget:
                break
            tokens += t
            i += 1
        windows.append((start, i))
    return windows


def _context_before(nodes: list[ASTNode], start: int, budget: int) -> str | None:
    """Markdown of the nodes just before `start`, in order, up to the token budget."""
    picked: list[str] = []
    tokens = 0
    for node in reversed(nodes[:start]):
        t = _est_tokens(node)
        if picked and tokens + t > budget:
            break
        tokens += t
        picked.append(node.content or "")
    return "\n\n".join(reversed(picked)) or None


def _context_after(nodes: list[ASTNode], end: int, budget: int) -> str | None:
    """Markdown of the nodes from `end` onward, in order, up to the token budget."""
    picked: list[str] = []
    tokens = 0
    for node in nodes[end:]:
        t = _est_tokens(node)
        if picked and tokens + t > budget:
            break
        tokens += t
        picked.append(node.content or "")
    return "\n\n".join(picked) or None


def _reconcile(entities: list[Entity]) -> list[Entity]:
    """Single-threaded post-pass over the window-ordered entity list: where a
    tail_open entity is immediately followed by a head_continuation entity, they are
    one entity split across a window cut — merge them. The opener (already anchored on
    its cue) keeps its type; the continuation's members append. Merging carries the
    continuation's tail_open forward so an entity spanning 3+ windows keeps absorbing."""
    out: list[Entity] = []
    for entity in entities:
        if out and out[-1].tail_open and entity.head_continuation:
            opener = out[-1]
            opener.members = opener.members + entity.members
            opener.tail_open = entity.tail_open
        else:
            out.append(entity)
    return out


class EntityGrouperNode:
    def __init__(
        self,
        module: Module | None = None,
        current_budget: int = CURRENT_BUDGET,
        prev_budget: int = PREV_BUDGET,
        next_budget: int = NEXT_BUDGET,
    ):
        self.module = module or Module()
        self.current_budget = current_budget
        self.prev_budget = prev_budget
        self.next_budget = next_budget

    def dispatch(self, state: State) -> list[Send] | str:
        """Cut the flat stream into dumb-greedy windows and fan out one worker each,
        carrying read-only prev/next context so edge entities can be flagged."""
        nodes = state.get("nodes", [])
        sends: list[Send] = []
        for wi, (start, end) in enumerate(_windows(nodes, self.current_budget)):
            sends.append(Send("entity_grouper_worker", {
                "window_index": wi,
                "current": nodes[start:end],
                "prev_context": _context_before(nodes, start, self.prev_budget),
                "next_context": _context_after(nodes, end, self.next_budget),
            }))
        return sends or "entity_grouper_collect"

    async def worker(self, state: dict) -> dict:
        """Gather definitions/theorems in one window; resolve local position spans to
        stable global node ids and emit provisional entities (with edge flags)."""
        current: list[ASTNode] = state["current"]
        window_nodes = [
            WindowNode(position=i, type=(n.type.value if n.type else ""), content=n.content)
            for i, n in enumerate(current)
        ]
        prediction = await self.module.aforward(
            current_nodes=window_nodes,
            previous_context=state.get("prev_context"),
            next_context=state.get("next_context"),
        )

        last = len(current) - 1
        entities: list[Entity] = []
        for span in prediction.entities:
            if span.type not in _GATHERED:
                continue
            # Clamp the LLM's positions into range and keep the span non-empty.
            start = min(max(span.start, 0), last)
            end = min(max(span.end, start), last)
            # Resolve to stable ids, dropping any atomic problem node that slipped into
            # the span — those are wrapped 1:1 separately and must not be double-counted.
            member_ids = [
                n.id for n in current[start:end + 1]
                if n.id is not None and n.type != NodeType.PROBLEM
            ]
            if not member_ids:
                continue
            entities.append(Entity(
                type=EntityType(span.type),
                members=[Member(node_id=i) for i in member_ids],
                head_continuation=bool(span.continues_before),
                tail_open=bool(span.continues_after),
            ))
        return {"entity_results": [(state["window_index"], entities)]}

    def collect(self, state: State) -> dict:
        """Assemble the sparse entity overlay: order the gathered def/thm entities by
        window, stitch cross-window spans, wrap problems 1:1, fold everything into one
        document-ordered list, and assign entity ids."""
        nodes = state.get("nodes", [])

        # 1. Window-ordered concatenation of the gathered def/thm entities.
        by_window = dict(state.get("entity_results", []))
        gathered: list[Entity] = []
        for wi in sorted(by_window):
            gathered.extend(by_window[wi])

        # 2. Stitch entities split across window cuts.
        gathered = _reconcile(gathered)

        # 3. Wrap each atomic problem node 1:1 (no LLM, no gathering).
        problems = [
            Entity(type=EntityType.PROBLEM, members=[Member(node_id=n.id)])
            for n in nodes if n.type == NodeType.PROBLEM and n.id is not None
        ]

        # 4. Fold both sources into one document-ordered list, keyed by each entity's
        #    first member's position in the stream, and assign stream-order ids.
        position = {n.id: i for i, n in enumerate(nodes)}
        combined = gathered + problems
        combined.sort(key=lambda e: position.get(e.members[0].node_id, len(nodes)) if e.members else len(nodes))
        for i, entity in enumerate(combined):
            entity.id = i
            entity.tail_open = False
            entity.head_continuation = False

        return {"entities": combined}
