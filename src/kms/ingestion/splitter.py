r"""
Exercise splitter — a post-seam node-normalizer that fixes exercise granularity
at the node level, so everything downstream stays simple.

The purely-structural extractor packs a run of exercises ("1.23 … 1.24 … 1.25
…") into ONE `list` node, because structurally it *is* one list. That coarse
node is the root cause of the exercise-entity granularity problem: the problem
finder can only point every exercise at the same node id, giving
indistinguishable duplicate entities. This stage removes the problem at its
source — it rewrites the canonical node stream so each exercise is its OWN node.
After it runs, the finder sees atomic exercises and emits one clean entity each,
with precise members; no downstream reconciliation is needed.

It does ONE job: SPLIT — any node that packs two or more numbered exercises is
replaced, in place, by one node per exercise (its reference number kept as
literal leading text, subparts kept nested, incidental markers like a "✓"
recommended glyph kept verbatim for provenance). A single isolated exercise or a
worked example is left untouched — only GROUPS are split. A shared-instruction
lead-in embedded between the exercises is broken out onto its own node too, so
it becomes atomic for the pedagogical component finder downstream.

Because the decision is per-node (a node either is or isn't a composite list)
and a node's content is always wholly inside some window, there is no
cross-window banking to get right: the walk gathers every split keyed by the
original node id, then rebuilds the stream once and re-assigns ids.
`segment_index` is inherited by each split piece, so picture resolution at
assembly is unaffected.

Wired in by ``SplitterNode`` (bottom of file): it runs right after the seam
merger, overwriting the `nodes` channel with the normalized stream.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state, walker

logger = logging.getLogger(__name__)

# Same look-ahead budget shape as the finders (~4 chars/token). A packed
# exercise list is one node and always fits whole (min-one-node), so the budget
# only needs to be large enough to hold that node; a single list is all one
# split call needs to see.
LOOKAHEAD_BUDGET = 2000


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: position, type, content."""

    position: int
    type: str
    content: str | None = None


class SplitExercise(BaseModel):
    """One piece carved out of a packed list node.

    Usually an exercise (its own number and text), but it may instead be a
    leading continuation fragment or an embedded lead-in (both with an EMPTY
    `number`), so nothing in the node is dropped.
    """

    number: str = Field(
        description="The exercise's own reference number as written, e.g. '1.23'. EMPTY for a leading continuation fragment that belongs to a previous exercise, or for an embedded shared-instruction lead-in."
    )
    content: str = Field(
        description="The piece's own text, copied verbatim, with its subparts, WITHOUT the leading number."
    )


class NodeSplit(BaseModel):
    """A node that packs two or more exercises, and its split pieces."""

    position: int = Field(
        description='The window position of the node that packs the exercises.'
    )
    exercises: list[SplitExercise] = Field(
        description='The individual exercises it holds, in order (two or more).'
    )


class Signature(dspy.Signature):
    r"""
    Normalise a run of textbook nodes for the exercise layer.

    SPLITS — find any single node that packs TWO OR MORE numbered exercises into
    one block (usually a `list` node like "1.23 … 1.24 … 1.25 …"). Return that
    node's position and its exercises IN ORDER, each with its own `number`
    ("1.23") and its own `content` (that exercise's statement text, copied
    VERBATIM — same wording, same LaTeX, same math, do not paraphrase, reflow,
    or drop any subpart — keeping its subparts (a)(b)(c) together and keeping
    any incidental markers like a leading "✓", but WITHOUT the reference
    number).

    PRESERVE A LEADING FRAGMENT: if the node BEGINS with text that belongs to a
    PREVIOUS exercise (a continuation the layout left at the top of this node —
    e.g. trailing subparts "(d) … (e) …" before the first numbered exercise
    here), return it as the FIRST item with an EMPTY `number` and that fragment
    as its verbatim `content`, so nothing is lost.

    BREAK OUT AN EMBEDDED LEAD-IN: if a piece BETWEEN the exercises is a
    shared-instruction lead-in (no number of its own, a directive for the run
    that follows it, e.g. "9-16 Sketch the polar curve."), return it as its OWN
    item with an EMPTY `number` and its verbatim text as `content`, so it lands
    on its own node as atomic input for the pedagogical component finder.

    Every character of the node must land in exactly one item, in order. A node
    holding only ONE exercise is NOT a split — leave it out. Worked examples,
    definitions, theorems, prose, and headers are never splits.

    Use the given `position` values, over the given nodes ONLY. The list may be
    empty.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    splits: list[NodeSplit] = dspy.OutputField(
        description='Nodes that pack two or more exercises, each split into its individual exercises.'
    )


class Decision(BaseModel):
    """The splitter's per-window verdict.

    Positions are already resolved to real node ids.
    """

    # Node id -> its exercise pieces.
    splits: dict[int, list[SplitExercise]] = {}


class Splitter(dspy.Module):
    """Splits nodes that pack multiple exercises into per-exercise nodes.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.splitter = dspy.ChainOfThought(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(
        self, current_nodes: list[WindowNode]
    ) -> list[NodeSplit]:
        """Judge one window.

        Args:
            current_nodes: The window's nodes, each with a local position.

        Returns:
            The window's split decisions.
        """
        result = await self.splitter.acall(current_nodes=current_nodes)
        splits = list(result.splits or [])
        # Whether a given packed node splits is the stage's known run-to-run
        # variance (docs/HANDOFF.md, known issues), so log the per-window
        # verdict.
        logger.debug(
            'split: %d nodes in, %d split(s) out%s',
            len(current_nodes),
            len(splits),
            ''.join(
                f' | position {split.position} -> '
                f'{len(split.exercises)} piece(s)'
                for split in splits
            ),
        )
        return splits

    def forward(self, current_nodes: list[WindowNode]) -> list[NodeSplit]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes))


async def _gather_decisions(
    nodes: list[models.ASTNode], module: Splitter, budget: int
) -> Decision:
    """Walk the stream in windows and collect every split, keyed by node id.

    A split is per-node (a node's content is wholly inside one window), so the
    cursor simply advances by the whole window — no banking, no growth.

    Args:
        nodes: The flat node stream.
        module: The splitting module.
        budget: The per-window soft token budget.

    Returns:
        Every split, keyed by the id of the node it replaces.
    """
    decision = Decision()
    cursor, node_count = 0, len(nodes)
    while cursor < node_count:
        end = walker.window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        splits = await module.aforward(
            [
                WindowNode(
                    position=position,
                    type=node.kind,
                    content=node.content,
                )
                for position, node in enumerate(window)
            ]
        )
        for split_result in splits:
            clamped = min(max(split_result.position, 0), last_local)
            node_id = window[clamped].id
            items = [
                exercise
                for exercise in split_result.exercises
                if (exercise.content or '').strip()
                or (exercise.number or '').strip()
            ]
            # Only a genuine group is a split.
            if node_id is not None and len(items) >= 2:
                decision.splits[node_id] = items
        cursor = end
    return decision


def _rebuild(
    nodes: list[models.ASTNode], decision: Decision
) -> list[models.ASTNode]:
    """Materialise the normalised stream.

    Replaces each split node with one node per piece, passes everything else
    through, and re-assigns ids. A split piece inherits its parent's `type` and
    `segment_index`; its content is the reference number as literal leading
    text followed by the exercise body, so the number survives and assembly
    stays faithful.

    Args:
        nodes: The original flat node stream.
        decision: The collected splits, keyed by node id.

    Returns:
        The rebuilt, re-id'd stream.
    """
    out: list[models.ASTNode] = []
    for node in nodes:
        pieces = decision.splits.get(node.id)
        if pieces:
            for item in pieces:
                number = (item.number or '').strip()
                body = (item.content or '').strip()
                content = f'{number} {body}'.strip() if number else body
                out.append(
                    type(node)(
                        content=content,
                        segment_index=node.segment_index,
                    )
                )
        else:
            out.append(node)
    for i, node in enumerate(out):
        node.id = i
    return out


async def split_exercises(
    nodes: list[models.ASTNode],
    module: Splitter | None = None,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[models.ASTNode]:
    """Split packed exercise nodes into per-exercise nodes.

    Args:
        nodes: The flat node stream.
        module: The splitting module. Created fresh if None.
        budget: The per-window soft token budget.

    Returns:
        A new, re-id'd node list (the canonical stream is mutated).
    """
    module = module or Splitter()
    if not nodes:
        return nodes
    decision = await _gather_decisions(nodes, module, budget)
    rebuilt = _rebuild(nodes, decision)
    logger.info(
        'splitter: %d node(s) -> %d (%d packed node(s) split)',
        len(nodes),
        len(rebuilt),
        len(decision.splits),
    )
    return rebuilt


# --- LangGraph node: normalise the node stream between the seam merger and the
# finders ---


class SplitterNode:
    """Rewrites `nodes` so each exercise and lead-in is its own node.

    A single sequential walk (a cursor over the stream cannot be sharded), so
    this is a plain graph node. It runs after the seam merger and before the
    instruction finder; overwriting the `nodes` channel is safe because no
    entity overlay exists yet — nothing references the old ids.

    Args:
        module: The splitting module. Created fresh if None.
    """

    def __init__(self, module: Splitter | None = None) -> None:
        self.module = module or Splitter()

    async def run(self, state: state.State) -> dict:
        """Normalise the node stream so each exercise is its own node.

        Args:
            state: The pipeline state, holding the flat node stream.

        Returns:
            The normalised `nodes` channel.
        """
        nodes = await split_exercises(
            state.get('nodes', []), module=self.module
        )
        return {'nodes': nodes}
