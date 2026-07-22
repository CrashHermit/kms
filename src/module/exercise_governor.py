r"""
Exercise governor — a finder-shaped window walk that splits grouped exercises and
propagates their shared instruction.

The Problem finder handles worked examples and standalone problems well, but a *run of
exercises* is a blind spot: the purely-structural extractor packs "1.23 … 1.24 … 1.25 …"
into ONE list node, so the finder can only point every one of them at that same node id —
indistinguishable pointers, duplicate entities. And a grouped exercise's imperative is
often stated ONCE in a lead-in ("In Exercises 1.23-1.25, find the eigenvalues of each
matrix.") that governs the whole run and is not a member of any individual problem.

Both are the same structure — a lead-in plus its exercise list — so one pass handles both.
It is shaped exactly like the three finders: a cursor takes a growing look-ahead window over
the node stream, and because it walks in order the lead-in and its list arrive in the SAME
window (no reaching backward by a fixed distance). From each grouped-exercise block the LLM
returns the shared instruction and the individual exercises (number + content); the walk
then emits one fine-grained Problem entity per exercise:

  * ``members`` — the block's node ids (coarse provenance: "these came from this list");
  * ``contents`` — that one exercise's text (distinct per entity, so the entities are
    distinguishable even though their members coincide);
  * ``number`` — that exercise's own number;
  * ``instruction`` — the block's shared directive (empty if the run has no lead-in — the
    split still applies; instruction propagation is the extra job when a lead-in exists).

Banking is the finders' structural rule: a block is banked only once a node is seen to
follow it (so a block can't be split by a window cut); if the sole block reaches the
window's edge the window grows and re-reads. Only GROUPS are this pass's concern (a run of
several exercises, typically one list node); isolated single problems remain the ordinary
Problem finder's job. Reconciling these fine-grained exercise entities with the finder's
coarse ones is a wiring step, done later — this pass just produces the split.
"""

import dspy
from pydantic import BaseModel, Field

from .state import ASTNode, Entity, EntityType
from .llm import text_lm

# Same look-ahead budget shape as the finders (~4 chars/token). A lead-in plus a long
# exercise list is the usual reason the window has to grow.
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""
    position: int
    type: str
    content: str | None = None


class ExerciseItem(BaseModel):
    """One exercise inside a grouped block: its own number and its own statement text."""
    number: str = Field(description="The exercise's own reference number, e.g. '1.23'.")
    content: str = Field(description="The exercise's own statement text, copied verbatim.")


class ExerciseBlock(BaseModel):
    """A grouped-exercise block: a span of window positions, the shared instruction (if any),
    and the individual exercises it contains."""
    start: int = Field(description="First window position of the block (its lead-in, or its list).")
    end: int = Field(description="Last window position of the block (inclusive).")
    instruction: str = Field(description="The shared directive from the lead-in, or empty string if none.")
    exercises: list[ExerciseItem] = Field(description="The individual exercises in the block, in order.")


class Signature(dspy.Signature):
    r"""
    Find GROUPED-EXERCISE blocks in a run of textbook nodes and split each into its
    individual exercises. A grouped-exercise block is a run of SEVERAL numbered exercises —
    usually packed into one list node ("1.23 ... 1.24 ... 1.25 ...") — optionally preceded by
    a LEAD-IN that states a shared instruction governing the whole run ("In Exercises
    1.23-1.25, find the eigenvalues of each matrix.").

    For each block return:
      * start, end — the inclusive window positions the block occupies (include the lead-in
        node in the span if there is one, through the last exercise node).
      * instruction — the shared imperative from the lead-in, copied as written ("find the
        eigenvalues of each matrix"). EMPTY STRING if the run has no shared lead-in (each
        exercise carries its own instruction).
      * exercises — the individual exercises IN ORDER, each with its own `number` (e.g.
        "1.23") and its own `content` (that one exercise's statement text, copied verbatim,
        WITHOUT the shared lead-in and without the leading number if it is separable).

    WHAT COUNTS: only GROUPS — a run of two or more exercises. A single isolated problem, a
    worked example, a definition, or a theorem is NOT this pass's concern; ignore them.
    Ordinary narrative prose, section headers, and figures are not exercise blocks.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values.
    - Return the blocks in document order.
    - If there are no grouped-exercise blocks in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    blocks: list[ExerciseBlock] = dspy.OutputField(
        description="The grouped-exercise blocks found, split into individual exercises, in document order."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[ExerciseBlock]:
        result = await self.finder.acall(current_nodes=current_nodes)
        return list(result.blocks or [])


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


def _entities_from_block(block: ExerciseBlock, window: list[ASTNode]) -> list[Entity]:
    """Turn one banked block into one fine-grained Problem entity per exercise: the block's
    node ids are shared coarse provenance; contents/number/instruction distinguish them."""
    ids = [window[k].id for k in range(block.start, block.end + 1) if window[k].id is not None]
    instruction = block.instruction.strip() or None
    out: list[Entity] = []
    for item in block.exercises:
        content = (item.content or "").strip()
        if not content:
            continue
        out.append(Entity(
            type=EntityType.PROBLEM,
            members=list(ids),
            contents=[content],
            number=(item.number.strip() or None),
            instruction=instruction,
        ))
    return out


async def govern_exercises(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[Entity]:
    """Cursor-walk the node stream and return fine-grained exercise Problem entities.

    Same growing-window banking rule as the finders: bank every block a node is seen to
    follow (bounded) and advance past it; if the only block reaches the window's edge, grow
    and re-read from the same cursor. Each banked block is split into one entity per exercise
    (shared instruction propagated). WindowNodes carry the real node ids so members resolve."""
    module = module or Module()
    exercises: list[Entity] = []
    cursor, n = 0, len(nodes)

    while cursor < n:
        size = budget
        while True:
            end = _window_from(nodes, cursor, size)
            window = nodes[cursor:end]  # ASTNodes, kept for id resolution
            last_local = len(window) - 1
            reached_doc_end = end == n

            blocks = await module.aforward([
                WindowNode(position=k, type=(node.type.value if node.type else ""), content=node.content)
                for k, node in enumerate(window)
            ])
            clean: list[ExerciseBlock] = []
            for b in blocks:
                start = min(max(b.start, 0), last_local)
                stop = min(max(b.end, start), last_local)
                if b.exercises:
                    clean.append(ExerciseBlock(start=start, end=stop, instruction=b.instruction, exercises=b.exercises))
            clean.sort(key=lambda b: b.start)

            if not clean:
                cursor = end  # no grouped exercises here — skip the window
                break

            bounded = [b for b in clean if b.end < last_local]

            if reached_doc_end or size >= max_budget:
                to_bank, advance = clean, end
            elif bounded:
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                size *= 2
                continue

            for b in to_bank:
                exercises.extend(_entities_from_block(b, window))
            cursor = advance
            break

    return exercises
