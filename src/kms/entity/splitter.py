r"""
Exercise splitter — a post-seam node-normalizer that fixes exercise granularity at the
node level, so everything downstream stays simple.

The purely-structural extractor packs a run of exercises ("1.23 … 1.24 … 1.25 …") into ONE
`list` node, because structurally it *is* one list. That coarse node is the root cause of the
exercise-entity granularity problem: the problem finder can only point every exercise at the
same node id, giving indistinguishable duplicate entities. This stage removes the problem at
its source — it rewrites the canonical node stream so each exercise is its OWN node. After it
runs, the finder sees atomic exercises and emits one clean entity each, with precise members;
no downstream reconciliation is needed.

It does ONE job: SPLIT — any node that packs two or more numbered exercises is replaced, in
place, by one node per exercise (its reference number kept as literal leading text, subparts
kept nested, incidental markers like a "✓" recommended glyph kept verbatim for provenance). A
single isolated exercise or a worked example is left untouched — only GROUPS are split. A
shared-instruction lead-in embedded between the exercises is broken out onto its own node too,
so it becomes atomic; TAGGING lead-ins is a separate stage (``instruction_finder``) that runs
right after this one, over the already-atomic stream.

Because the decision is per-node (a node either is or isn't a composite list) and a node's
content is always wholly inside some window, there is no cross-window banking to get right: the
walk gathers every split keyed by the original node id, then rebuilds the stream once and
re-assigns ids. `seg_index` is inherited by each split piece, so picture resolution at assembly
is unaffected.

Wired in by ``SplitterNode`` (bottom of file): it runs right after the seam merger and before
the instruction finder, overwriting the `nodes` channel with the normalized stream.
"""

import os
from pathlib import Path

import dspy
from pydantic import BaseModel, Field

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import ASTNode
from kms.core.state import State

# Same look-ahead budget shape as the finders (~4 chars/token). A packed exercise list is one
# node and always fits whole (min-one-node), so the budget only needs to be large enough to
# hold that node; a single list is all one split call needs to see.
LOOKAHEAD_BUDGET = 2000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class SplitExercise(BaseModel):
    """One piece carved out of a packed list node: usually an exercise (its own number and
    text), but it may instead be a leading continuation fragment or an embedded lead-in
    (both with an EMPTY `number`), so nothing in the node is dropped."""

    number: str = Field(
        description="The exercise's own reference number as written, e.g. '1.23'. EMPTY for a leading continuation fragment that belongs to a previous exercise, or for an embedded shared-instruction lead-in."
    )
    content: str = Field(
        description="The piece's own text, copied verbatim, with its subparts, WITHOUT the leading number."
    )


class NodeSplit(BaseModel):
    """A single node that packs two or more numbered exercises, and its split pieces."""

    position: int = Field(description="The window position of the node that packs the exercises.")
    exercises: list[SplitExercise] = Field(
        description="The individual exercises it holds, in order (two or more)."
    )


class Signature(dspy.Signature):
    r"""
    Normalise a run of textbook nodes for the exercise layer.

    SPLITS — find any single node that packs TWO OR MORE numbered exercises into one block
    (usually a `list` node like "1.23 … 1.24 … 1.25 …"). Return that node's position and its
    exercises IN ORDER, each with its own `number` ("1.23") and its own `content` (that
    exercise's statement text, copied VERBATIM — same wording, same LaTeX, same math, do not
    paraphrase, reflow, or drop any subpart — keeping its subparts (a)(b)(c) together and
    keeping any incidental markers like a leading "✓", but WITHOUT the reference number).

    PRESERVE A LEADING FRAGMENT: if the node BEGINS with text that belongs to a PREVIOUS
    exercise (a continuation the layout left at the top of this node — e.g. trailing subparts
    "(d) … (e) …" before the first numbered exercise here), return it as the FIRST item with an
    EMPTY `number` and that fragment as its verbatim `content`, so nothing is lost.

    BREAK OUT AN EMBEDDED LEAD-IN: if a piece BETWEEN the exercises is a shared-instruction
    lead-in (no number of its own, a directive for the run that follows it, e.g. "9-16 Sketch
    the polar curve."), return it as its OWN item with an EMPTY `number` and its verbatim text as
    `content`, so it lands on its own node. Do NOT tag it and do NOT fold it into an exercise —
    a later pass decides which nodes are lead-ins.

    Every character of the node must land in exactly one item, in order. A node holding only ONE
    exercise is NOT a split — leave it out. Worked examples, definitions, theorems, prose, and
    headers are never splits.

    Use the given `position` values, over the given nodes ONLY. The list may be empty.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    splits: list[NodeSplit] = dspy.OutputField(
        description="Nodes that pack two or more exercises, each split into its individual exercises."
    )


class Decision(BaseModel):
    """The splitter's per-window verdict, positions already resolved to real node ids."""

    splits: dict[int, list[SplitExercise]] = {}  # node id -> its exercise pieces


# Compiled few-shot program produced by `training/splitter/compile.py`. Loaded at serve
# time if present so the data-optimized demonstrations ship with the pipeline; override the
# path with KMS_SPLITTER_PROGRAM, or delete the file to fall back to the bare student.
_COMPILED_PATH = os.environ.get("KMS_SPLITTER_PROGRAM", "training/splitter/compiled.json")


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None, compiled: bool = True) -> None:
        super().__init__()
        self.splitter = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())
        if compiled and Path(_COMPILED_PATH).exists():
            self.load(_COMPILED_PATH)

    def forward(self, current_nodes: list[WindowNode]) -> dspy.Prediction:
        """Synchronous predictor pass — used at DSPy compile/eval time (optimizers and the
        judge run modules synchronously). Serving uses ``aforward``; both share the one
        ``self.splitter`` predictor, so demonstrations compiled here transfer to serving."""
        return self.splitter(current_nodes=current_nodes)

    async def aforward(self, current_nodes: list[WindowNode]) -> list[NodeSplit]:
        result = await self.splitter.acall(current_nodes=current_nodes)
        splits = list(result.splits or [])
        tracing.record(
            "splitter",
            inputs={"current_nodes": [n.model_dump() for n in current_nodes]},
            outputs={"splits": [s.model_dump() for s in splits]},
        )
        return splits


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


async def _gather_decisions(nodes: list[ASTNode], module: Module, budget: int) -> Decision:
    """Walk the stream in windows and collect every split, keyed by node id.

    A split is per-node (a node's content is wholly inside one window), so the cursor simply
    advances by the whole window — no banking, no growth."""
    decision = Decision()
    cursor, n = 0, len(nodes)
    while cursor < n:
        end = _window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        splits = await module.aforward(
            [
                WindowNode(
                    position=k, type=(node.type.value if node.type else ""), content=node.content
                )
                for k, node in enumerate(window)
            ]
        )
        for s in splits:
            p = min(max(s.position, 0), last_local)
            nid = window[p].id
            items = [
                e for e in s.exercises if (e.content or "").strip() or (e.number or "").strip()
            ]
            if nid is not None and len(items) >= 2:  # only a genuine group is a split
                decision.splits[nid] = items
        cursor = end
    return decision


def _rebuild(nodes: list[ASTNode], decision: Decision) -> list[ASTNode]:
    """Materialise the normalised stream: replace each split node with one node per piece,
    pass everything else through, and re-assign ids.

    A split piece inherits its parent's `type` and `seg_index`; its content is the reference
    number as literal leading text followed by the exercise body, so the number survives and
    assembly stays faithful."""
    out: list[ASTNode] = []
    for node in nodes:
        pieces = decision.splits.get(node.id)
        if pieces:
            for item in pieces:
                number = (item.number or "").strip()
                body = (item.content or "").strip()
                content = f"{number} {body}".strip() if number else body
                out.append(ASTNode(type=node.type, content=content, seg_index=node.seg_index))
        else:
            out.append(node)
    for i, node in enumerate(out):
        node.id = i
    return out


async def split_exercises(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[ASTNode]:
    """Normalise the flat node stream: split packed exercise nodes into per-exercise nodes.
    Returns a new, re-id'd node list (the canonical stream is mutated)."""
    module = module or Module()
    if not nodes:
        return nodes
    decision = await _gather_decisions(nodes, module, budget)
    return _rebuild(nodes, decision)


# --- LangGraph node: normalise the node stream between the seam merger and the finders ---


class SplitterNode:
    """Rewrites the `nodes` channel so each exercise (and each embedded lead-in) is its own node.

    A single sequential walk (a cursor over the stream cannot be sharded), so this is a plain
    graph node. It runs after the seam merger and before the instruction finder; overwriting the
    `nodes` channel is safe because no entity overlay exists yet — nothing references the old
    ids."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        nodes = await split_exercises(state.get("nodes", []), module=self.module)
        return {"nodes": nodes}
