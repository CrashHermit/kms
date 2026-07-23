r"""
Instruction finder — a cursor-walk that tags exercise LEAD-IN nodes `role="instruction"`.

It runs right after the exercise splitter, over the already-atomic node stream. Because the
splitter has broken every packed list (and every embedded lead-in) onto its own node, every
lead-in is now a standalone node — so tagging is ONE uniform decision per node: "is this node
a shared-instruction lead-in?" (There is deliberately no split/segment work here; the splitter
owns that.) The tags are consumed later by the instruction distributor, which copies each
lead-in's directive onto the Problems it governs.

A lead-in is a directive with NO reference number of its own that introduces a run of
separately-numbered exercises ("In Exercises 1.23-1.25, find the eigenvalues …", "For the
following exercises, find the gradient."). A node that begins with its OWN exercise number is
an exercise, never a lead-in — that distinction is the decisive test in the prompt (it killed 8
false positives on a Hefferon section with zero true lead-ins).

The walk is a plain window cursor: read a window of whole nodes up to a soft token budget, ask
the LLM which positions are lead-ins, stamp `role="instruction"` on those nodes, advance by the
whole window. The decision is per-node (a node either is or isn't a lead-in) and a node lives
wholly inside one window, so there is no cross-window banking. Its output is tiny (a list of
positions), so unlike the splitter it is immune to output-token truncation.

Wired in by ``InstructionFinderNode`` (bottom of file): it rewrites the `nodes` channel with
the tagged stream, between the splitter and the node persister.
"""

import dspy
from pydantic import BaseModel

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import ASTNode
from kms.core.state import State

# Same look-ahead budget shape as the finders (~4 chars/token). A lead-in and the exercise it
# introduces are small; the budget only needs enough context to tell a lead-in from an exercise.
LOOKAHEAD_BUDGET = 2000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Signature(dspy.Signature):
    r"""
    Find the exercise LEAD-IN nodes in a run of textbook nodes and return their positions.

    A lead-in is a short directive that introduces a run of OTHER exercises and states a shared
    instruction. The decisive test: a lead-in has NO reference number of its own, yet gives an
    imperative meant for a run of separately-numbered exercises that follow it. It may name an
    explicit RANGE ("In Exercises 1.23-1.25, find the eigenvalues of each matrix."; "For
    Exercises 5-6, determine whether the set is a subspace.") OR name no range at all — and the
    range-less form is the COMMON one, so do NOT require a range: "In the following exercises,
    simplify each expression.", "For the following exercises, find the gradient.", "Prove each of
    the following.", "9-16 Sketch the polar curve." are all lead-ins. Tag either form.

    CRITICAL: a node that BEGINS WITH ITS OWN EXERCISE NUMBER ("1.15 Perform each
    multiplication.", "✓ 1.17 For a homomorphism …", "1.22 Represent each linear map …") is an
    EXERCISE, never a lead-in — do NOT tag it, even though its own imperative reads like an
    instruction, because that imperative governs only that one exercise's own subparts. A lead-in
    governs several DIFFERENTLY-numbered exercises; an exercise governs only its own (a)(b)(c).
    Prose and section headers are never lead-ins either.

    Return the positions of the lead-in nodes, over the given nodes ONLY. The list may be empty.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    instruction_positions: list[int] = dspy.OutputField(
        description="Positions of exercise lead-in nodes (shared-instruction directives)."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, current_nodes: list[WindowNode]) -> list[int]:
        result = await self.finder.acall(current_nodes=current_nodes)
        positions = list(result.instruction_positions or [])
        tracing.record(
            "instruction_finder",
            inputs={"current_nodes": [n.model_dump() for n in current_nodes]},
            outputs={"instruction_positions": positions},
        )
        return positions


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


async def tag_instructions(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[ASTNode]:
    """Walk the stream in windows and stamp `role="instruction"` on every lead-in node, in
    place. Returns the same node list (mutated)."""
    module = module or Module()
    if not nodes:
        return nodes
    cursor, n = 0, len(nodes)
    while cursor < n:
        end = _window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        positions = await module.aforward(
            [
                WindowNode(
                    position=k, type=(node.type.value if node.type else ""), content=node.content
                )
                for k, node in enumerate(window)
            ]
        )
        for pos in positions:
            p = min(max(pos, 0), last_local)
            window[p].role = "instruction"
        cursor = end
    return nodes


# --- LangGraph node: tag lead-in nodes between the splitter and the node persister ---


class InstructionFinderNode:
    """Tags exercise lead-in nodes `role="instruction"` on the `nodes` channel.

    A single sequential walk (a cursor over the stream cannot be sharded), so this is a plain
    graph node. It runs after the splitter (so every lead-in is already its own node) and before
    the node persister and the three finders, which read the tags."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        nodes = await tag_instructions(state.get("nodes", []), module=self.module)
        return {"nodes": nodes}
