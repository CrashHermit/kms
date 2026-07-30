"""Seam merger — heals structural nodes split across adjacent page boundaries.

When a single block (paragraph, equation, list item) spans two OCR pages, the
extractor produces an incomplete tail on the top page and an incomplete head on
the bottom page. This stage uses a DSPy ChainOfThought module to decide whether
each cross-page pair is a split block and merges the halves. Workers run in two
passes (even/odd) to avoid races on shared segments.
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

    merged: str | None = dspy.OutputField(
        description='The merged content. If the two edge nodes are split halves of the same block, return the combined text (tail content immediately followed by head content, joined naturally without an added separator). If they are already complete independent nodes, return None.'
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
    ) -> str | None:
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
        return result.merged

    def forward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> str | None:
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


def _pairs(
    segments: list[models.Segment], parity: int
) -> list[tuple[models.Segment, models.Segment]]:
    """The adjacent segment pairs one parity pass may fan out over.

    Args:
        segments: The ordered segment backbone.
        parity: 0 for the even pass (0-1, 2-3, …), 1 for the odd pass.

    Returns:
        The ``(top, bottom)`` pairs whose top index has the given parity and
        where both sides carry nodes.
    """
    return [
        (segments[i], segments[i + 1])
        for i in range(len(segments) - 1)
        if segments[i].index % 2 == parity
        and segments[i].nodes
        and segments[i + 1].nodes
    ]


async def _merge_pair(
    module: SeamMerger, top: models.Segment, bottom: models.Segment
) -> list[tuple[int, list[models.ASTNode]]]:
    """Merge one seam, if the LLM judges it to be a split node.

    A healed seam folds the merged content into the top's tail and drops the
    bottom's head.

    Args:
        module: The seam-merging module.
        top: The upper segment of the pair.
        bottom: The lower segment of the pair.

    Returns:
        Both segments' ``(segment_index, nodes)`` entries.
    """
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)

    tail = top_nodes[-1]
    head = bottom_nodes[0]
    top_context = top_nodes[-2] if len(top_nodes) > 1 else None
    bottom_context = bottom_nodes[1] if len(bottom_nodes) > 1 else None

    merged = await module.aforward(
        top_bottom_edge_node=_to_seam_node_dto(tail),
        bottom_top_edge_node=_to_seam_node_dto(head),
        top_node_context=_to_seam_node_dto(top_context),
        bottom_node_context=_to_seam_node_dto(bottom_context),
    )
    healed = merged is not None and bool(merged)
    logger.debug(
        'seam %d/%d: %s | tail %r + head %r',
        top.index,
        bottom.index,
        'merged' if healed else 'left split',
        logs.elide(tail.content, 40),
        logs.elide(head.content, 40),
    )
    if healed:
        tail.content = merged
        bottom_nodes = bottom_nodes[1:]

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
