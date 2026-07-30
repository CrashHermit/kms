"""Seam merger — heals structural nodes split across adjacent page boundaries.

When a single block (paragraph, equation, list item) spans two OCR pages, the
extractor produces an incomplete tail on the top page and an incomplete head on
the bottom page. This stage uses a DSPy ChainOfThought module to decide whether
each cross-page pair is a split block and merges the halves. Workers run in two
passes (even/odd) to avoid races on shared segments.

**Page apparatus — bibliographic references and notes — is not a seam candidate
and is skipped when choosing the edges.** The stage rests on an adjacency
assumption: that the last block of one page is the one that continues onto the
next. Apparatus breaks it in both directions.

A footnote arrives at the foot of its page (the front end appends the extracted
footer there), so it *displaces* the real tail — the paragraph that actually
runs onto the next page is no longer last. And apparatus is self-contained: a
reference list is a run of separate works with no continuation between them, and
a note hangs off a marker rather than off its neighbours, so a merge across such
a seam does not heal a split block, it welds two independent things into one
node that nothing downstream can separate again.

Skipping the *nodes* rather than the whole seam keeps the heal working on the
pages that have footnotes: the edges become the last and first mergeable blocks,
and the apparatus is simply passed over. The trade is that apparatus genuinely
split across a page break — a long footnote continued overleaf — now stays
split, which is the cheaper of the two errors: a dented node rather than two
unrelated things fused.
"""

import asyncio
import logging

import dspy
from langgraph.types import Send
from pydantic import BaseModel

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)


class SeamNodeDTO(BaseModel):
    """Lightweight DSPy boundary model: a node's content and type."""

    content: str | None = None
    types: list[str] = []


class Signature(dspy.Signature):
    """
    You are an expert technical editor. Two adjacent runs of document blocks
    share a seam — the boundary where one run ends and the next begins.
    Sometimes a single block (paragraph, sentence, equation, list item, caption,
    etc.) is split across that boundary, producing an incomplete tail in the top
    run and an incomplete head in the bottom run.

    Your job: decide whether the tail node of the top run and the head node of
    the bottom run are two halves of the same interrupted block. If they are,
    merge them into one coherent node. If they are not — they are already
    complete, independent nodes that merely sit next to each other at the
    boundary — return None.

    Judge this purely on structure: does the tail read as cut off mid-block and
    the head as its continuation? Do not reason about the subject matter or
    reassemble blocks that are each already complete.

    Use the context nodes (the neighbour just inside each run) only to inform
    your judgment — never include their content in the merged output.

    LATEX FORMAT: All mathematical notation must use LaTeX format. Use single
    dollar signs `$ $` for inline math and double dollar signs `$$ $$` for
    block/display math. Preserve existing delimiters and math content exactly.
    """

    top_node_context: SeamNodeDTO | None = dspy.InputField(
        description='The node immediately before the tail of the top element run. Read-only context — do not include its content in the output.'
    )
    top_bottom_edge_node: SeamNodeDTO = dspy.InputField(
        description='The tail node of the top element run — the candidate for merging.'
    )
    bottom_top_edge_node: SeamNodeDTO = dspy.InputField(
        description='The head node of the bottom element run — the other candidate for merging.'
    )
    bottom_node_context: SeamNodeDTO | None = dspy.InputField(
        description='The node immediately after the head of the bottom element run. Read-only context — do not include its content in the output.'
    )

    node: SeamNodeDTO | None = dspy.OutputField(
        description='The merged result. If the two edge nodes are split halves of the same node, return a single merged node combining their content. If they are already complete independent nodes, return None.'
    )


class SeamMerger(dspy.Module):
    """Decides whether two adjacent edge nodes are halves of one block.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.merger = dspy.ChainOfThought(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> SeamNodeDTO | None:
        """Judge one seam.

        Args:
            top_bottom_edge_node: The tail node of the top run.
            bottom_top_edge_node: The head node of the bottom run.
            top_node_context: The neighbour just inside the top run.
            bottom_node_context: The neighbour just inside the bottom run.

        Returns:
            The merged node if the pair is one split block, else None.
        """
        result = await self.merger.acall(
            top_node_context=top_node_context,
            top_bottom_edge_node=top_bottom_edge_node,
            bottom_top_edge_node=bottom_top_edge_node,
            bottom_node_context=bottom_node_context,
        )
        return result.node

    def forward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> SeamNodeDTO | None:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                top_bottom_edge_node=top_bottom_edge_node,
                bottom_top_edge_node=bottom_top_edge_node,
                top_node_context=top_node_context,
                bottom_node_context=bottom_node_context,
            )
        )


# --- LangGraph node: stitch nodes split across segment boundaries ---
#
# A worker touches two adjacent segments (top tail + bottom head), so adjacent
# pairs cannot run at once without racing on the shared segment. We run two
# passes: the even pass handles pairs whose top index is even (0-1, 2-3, ...),
# the odd pass handles the rest (1-2, 3-4, ...). Within a pass no two pairs
# share a segment, so they fan out safely; the passes run sequentially (even ->
# collect -> odd -> collect), and each pass writes its own reducer channel to
# avoid cross-pass contamination.


def _to_seam_node_dto(node: models.ASTNode | None) -> SeamNodeDTO:
    """The boundary model for one node.

    Args:
        node: The node to describe, or None for a missing neighbour.

    Returns:
        The boundary model, empty when no node was given.
    """
    if node is None:
        return SeamNodeDTO(content=None, types=[])
    return SeamNodeDTO(content=node.content, types=[node.kind])


# Node types that are never one half of a block split across a page break.
# Both sit outside the body flow — a reference names a work, a note hangs off a
# marker — so neither continues into its neighbour (see the module docstring).
_APPARATUS = (models.BibliographicNode, models.NoteNode)


def _mergeable_indices(nodes: list[models.ASTNode]) -> list[int]:
    """The positions of the nodes a seam may consider, in order.

    Page apparatus is passed over: it is never one half of a block split
    across a page break, and merging one would weld it onto its neighbour
    irreversibly (see the module docstring).

    Args:
        nodes: One segment's nodes, in document order.

    Returns:
        The indices of the mergeable nodes.
    """
    return [
        index
        for index, node in enumerate(nodes)
        if not isinstance(node, _APPARATUS)
    ]


def _pairs(
    segments: list[models.Segment], parity: int
) -> list[tuple[models.Segment, models.Segment]]:
    """The adjacent segment pairs one parity pass may fan out over.

    Args:
        segments: The ordered segment backbone.
        parity: 0 for the even pass (0-1, 2-3, …), 1 for the odd pass.

    Returns:
        The ``(top, bottom)`` pairs whose top index has the given parity and
        where both sides carry a mergeable node — a page whose only nodes are
        apparatus has no seam to heal, so no worker is spawned for it.
    """
    return [
        (segments[i], segments[i + 1])
        for i in range(len(segments) - 1)
        if segments[i].index % 2 == parity
        and _mergeable_indices(segments[i].nodes)
        and _mergeable_indices(segments[i + 1].nodes)
    ]


async def _merge_pair(
    module: SeamMerger, top: models.Segment, bottom: models.Segment
) -> list[tuple[int, list[models.ASTNode]]]:
    """Merge one seam, if the LLM judges it to be a split node.

    The edges are the top's last and the bottom's first *mergeable* nodes —
    page apparatus is passed over on both sides, so a page that ends in a
    footnote still has its real tail healed. A healed seam folds the merged
    content into that tail and drops that head.

    Args:
        module: The seam-merging module.
        top: The upper segment of the pair.
        bottom: The lower segment of the pair.

    Returns:
        Both segments' ``(segment_index, nodes)`` entries.
    """
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)

    top_mergeable = _mergeable_indices(top_nodes)
    bottom_mergeable = _mergeable_indices(bottom_nodes)
    if not top_mergeable or not bottom_mergeable:
        # Nothing on one side but apparatus: no seam to judge.
        return [(top.index, top_nodes), (bottom.index, bottom_nodes)]

    tail_index = top_mergeable[-1]
    head_index = bottom_mergeable[0]
    tail = top_nodes[tail_index]
    head = bottom_nodes[head_index]
    top_context = (
        top_nodes[top_mergeable[-2]] if len(top_mergeable) > 1 else None
    )
    bottom_context = (
        bottom_nodes[bottom_mergeable[1]] if len(bottom_mergeable) > 1 else None
    )

    merged = await module.aforward(
        top_bottom_edge_node=_to_seam_node_dto(tail),
        bottom_top_edge_node=_to_seam_node_dto(head),
        top_node_context=_to_seam_node_dto(top_context),
        bottom_node_context=_to_seam_node_dto(bottom_context),
    )
    healed = merged is not None and bool(merged.content)
    logger.debug(
        'seam %d/%d: %s | tail %r + head %r',
        top.index,
        bottom.index,
        'merged' if healed else 'left split',
        logs.elide(tail.content, 40),
        logs.elide(head.content, 40),
    )
    if healed:
        tail.content = merged.content
        del bottom_nodes[head_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class SeamMergerNode:
    """Heals cross-page splits using two parity passes to avoid races.

    Args:
        module: The seam-merging module. Created fresh if None.
    """

    def __init__(self, module: SeamMerger | None = None) -> None:
        self.module = module or SeamMerger()

    def dispatch_even(self, state: state.State) -> list[Send] | str:
        """Fans out workers for even-indexed segment pairs (0-1, 2-3, …)."""
        pairs = _pairs(state.get('segments', []), parity=0)
        sends = [
            Send('seam_even_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_even_collect'

    def dispatch_odd(self, state: state.State) -> list[Send] | str:
        """Fans out workers for odd-indexed segment pairs (1-2, 3-4, …)."""
        pairs = _pairs(state.get('segments', []), parity=1)
        sends = [
            Send('seam_odd_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_odd_collect'

    async def even_worker(self, state: dict) -> dict:
        """Merges one even pair and returns the healed segment nodes."""
        return {
            'seam_even_results': await _merge_pair(
                self.module, state['top'], state['bottom']
            )
        }

    async def odd_worker(self, state: dict) -> dict:
        """Merges one odd pair and returns the healed segment nodes."""
        return {
            'seam_odd_results': await _merge_pair(
                self.module, state['top'], state['bottom']
            )
        }

    def _collect(self, state: state.State, channel: str) -> dict:
        """Drain one pass's channel back into the segment backbone.

        Args:
            state: The pipeline state.
            channel: The reducer channel this pass wrote.

        Returns:
            The updated segment backbone.
        """
        segments = models.merge_results_into_segments(
            state['segments'], state.get(channel, []), 'nodes'
        )
        return {'segments': segments}

    def even_collect(self, state: state.State) -> dict:
        """Drains the even-pass results back into the segment backbone."""
        return self._collect(state, 'seam_even_results')

    def odd_collect(self, state: state.State) -> dict:
        """Drain the odd pass, then birth the flat global node list.

        The seam merger is the last stage that splits/merges nodes
        structurally, so page-splits are now healed and node identity is
        stable — flatten the per-page backbone into `nodes`, stamping each with
        its global id and originating segment_index. Every stage after this
        works on `nodes`, not on the per-segment nesting.

        Args:
            state: The pipeline state.

        Returns:
            The healed segment backbone and the flat node stream.
        """
        result = self._collect(state, 'seam_odd_results')
        segments = result['segments']
        nodes = models.flatten_segments(segments)
        # The handover between the pipeline's two phases: per-page segments
        # become one flat, stably-id'd stream that every later stage walks.
        logger.info(
            'seam merger: %d page(s) -> flat stream of %d node(s)',
            len(segments),
            len(nodes),
        )
        return {
            'segments': segments,
            'nodes': nodes,
        }
