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

It does two local jobs in one LLM walk over the flat stream:

  1. SPLIT — any node that packs two or more numbered exercises is replaced, in place, by one
     node per exercise (its reference number kept as literal leading text, subparts kept
     nested, incidental markers like a "✓" recommended glyph kept verbatim for provenance).
     A single isolated exercise or a worked example is left untouched — only GROUPS are split.
  2. TAG — a node that is an exercise LEAD-IN ("In Exercises 1.23-1.25, find the eigenvalues
     …") is annotated `role="instruction"`. The splitter does NOT try to work out which
     exercises the lead-in governs — that positional/range reasoning is the instruction
     distributor's job, later. Here the lead-in is only *marked*; its content is untouched.

Because both decisions are per-node (a node either is or isn't a composite list; is or isn't
a lead-in) and a node's content is always wholly inside some window, there is no cross-window
banking to get right: the walk gathers every decision keyed by the original node id, then
rebuilds the stream once and re-assigns ids. `seg_index` is inherited by each split piece, so
picture resolution at assembly is unaffected.

Wired in by ``SplitterNode`` (bottom of file): it runs right after the seam merger and before
the three finders, overwriting the `nodes` channel with the normalized stream.
"""

import os
from pathlib import Path

import dspy
from pydantic import BaseModel, Field

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import ASTNode
from kms.core.state import State

# Same look-ahead budget shape as the finders (~4 chars/token). The splitter only needs
# enough context to recognise a lead-in and the exercise run it introduces; a packed
# exercise list is one node and always fits whole.
LOOKAHEAD_BUDGET = 2000


def _est_tokens(node: ASTNode) -> int:
    return len(node.content or "") // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class SplitExercise(BaseModel):
    """One exercise carved out of a packed list node: its own number and its own text."""

    number: str = Field(
        description="The exercise's own reference number as written, e.g. '1.23'. EMPTY for a leading continuation fragment that belongs to a previous exercise."
    )
    content: str = Field(
        description="The exercise's own statement text, copied verbatim, with its subparts, WITHOUT the leading number."
    )


class NodeSplit(BaseModel):
    """A single node that packs two or more numbered exercises, and its split pieces."""

    position: int = Field(description="The window position of the node that packs the exercises.")
    exercises: list[SplitExercise] = Field(
        description="The individual exercises it holds, in order (two or more)."
    )


class Signature(dspy.Signature):
    r"""
    Normalise a run of textbook nodes for the exercise layer. Do TWO independent things:

    1. SPLITS — find any single node that packs TWO OR MORE numbered exercises into one block
       (usually a `list` node like "1.23 … 1.24 … 1.25 …"). Return that node's position and
       its exercises IN ORDER, each with its own `number` ("1.23") and its own `content` (that
       exercise's statement text, copied VERBATIM — same wording, same LaTeX, same math, do not
       paraphrase, reflow, or drop any subpart — keeping its subparts (a)(b)(c) together and
       keeping any incidental markers like a leading "✓", but WITHOUT the reference number).

       PRESERVE A LEADING FRAGMENT: if the node BEGINS with text that belongs to a PREVIOUS
       exercise (a continuation the layout left at the top of this node — e.g. trailing
       subparts "(d) … (e) …" before the first numbered exercise here), return it as the FIRST
       item with an EMPTY `number` and that fragment as its verbatim `content`, so nothing is
       lost. Every character of the node must land in exactly one item, in order.

       A node holding only ONE exercise is NOT a split — leave it out. Worked examples,
       definitions, theorems, prose, headers are never splits.

    2. INSTRUCTION_POSITIONS — find any node that is an exercise LEAD-IN: a short directive
       that introduces a run of OTHER exercises and states a shared instruction. The decisive
       test: a lead-in has NO reference number of its own, yet gives an imperative meant for a
       run of separately-numbered exercises that follow it. It may name an explicit RANGE
       ("In Exercises 1.23-1.25, find the eigenvalues of each matrix."; "For Exercises 5-6,
       determine whether the set is a subspace.") OR name no range at all — and the range-less
       form is the COMMON one, so do NOT require a range: "In the following exercises, simplify
       each expression.", "For the following exercises, find the gradient.", "Prove each of the
       following." are all lead-ins. Tag either form. Return its position.

       CRITICAL: a node that BEGINS WITH ITS OWN EXERCISE NUMBER ("1.15 Perform each
       multiplication.", "✓ 1.17 For a homomorphism …", "1.22 Represent each linear map …") is
       an EXERCISE, never a lead-in — do NOT tag it, even though its own imperative reads like
       an instruction, because that imperative governs only that one exercise's own subparts.
       A lead-in governs several DIFFERENTLY-numbered exercises; an exercise governs only its
       own (a)(b)(c). Prose and section headers are never lead-ins either.

    Use the given `position` values, over the given nodes ONLY. Both lists may be empty.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    splits: list[NodeSplit] = dspy.OutputField(
        description="Nodes that pack two or more exercises, each split into its individual exercises."
    )
    instruction_positions: list[int] = dspy.OutputField(
        description="Positions of exercise lead-in nodes (shared-instruction directives)."
    )


class Decision(BaseModel):
    """The splitter's per-window verdict, positions already resolved to real node ids."""

    splits: dict[int, list[SplitExercise]] = {}  # node id -> its exercise pieces
    instructions: set[int] = set()  # node ids that are lead-ins


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

    async def aforward(self, current_nodes: list[WindowNode]) -> tuple[list[NodeSplit], list[int]]:
        result = await self.splitter.acall(current_nodes=current_nodes)
        splits, instruction_positions = (
            list(result.splits or []),
            list(result.instruction_positions or []),
        )
        tracing.record(
            "splitter",
            inputs={"current_nodes": [n.model_dump() for n in current_nodes]},
            outputs={
                "splits": [s.model_dump() for s in splits],
                "instruction_positions": instruction_positions,
            },
        )
        return splits, instruction_positions


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
    """Walk the stream in windows and collect every per-node decision, keyed by node id.

    Per-node decisions can't be split across a window (a node's content is wholly inside one
    window), so the cursor simply advances by the whole window — no banking, no growth."""
    decision = Decision()
    cursor, n = 0, len(nodes)
    while cursor < n:
        end = _window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        splits, instruction_positions = await module.aforward(
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
        for pos in instruction_positions:
            p = min(max(pos, 0), last_local)
            nid = window[p].id
            if nid is not None:
                decision.instructions.add(nid)
        cursor = end
    return decision


def _rebuild(nodes: list[ASTNode], decision: Decision) -> list[ASTNode]:
    """Materialise the normalised stream: replace each split node with one node per exercise,
    tag each lead-in `role="instruction"`, pass everything else through, and re-assign ids.

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
            if node.id in decision.instructions:
                node.role = "instruction"
            out.append(node)
    for i, node in enumerate(out):
        node.id = i
    return out


async def split_exercises(
    nodes: list[ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[ASTNode]:
    """Normalise the flat node stream: split packed exercise nodes into per-exercise nodes and
    tag lead-in nodes. Returns a new, re-id'd node list (the canonical stream is mutated)."""
    module = module or Module()
    if not nodes:
        return nodes
    decision = await _gather_decisions(nodes, module, budget)
    return _rebuild(nodes, decision)


# --- LangGraph node: normalise the node stream between the seam merger and the finders ---


class SplitterNode:
    """Rewrites the `nodes` channel so each exercise is its own node and lead-ins are tagged.

    A single sequential walk (a cursor over the stream cannot be sharded), so this is a plain
    graph node. It runs after the seam merger and before the three finders; overwriting the
    `nodes` channel is safe because no entity overlay exists yet — nothing references the old
    ids."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        nodes = await split_exercises(state.get("nodes", []), module=self.module)
        return {"nodes": nodes}
