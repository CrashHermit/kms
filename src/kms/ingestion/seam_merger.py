"""Seam merger — heals structural nodes split across adjacent page boundaries.

When a single block (paragraph, equation, list item) spans two OCR pages, the
extractor produces an incomplete tail on the top page and an incomplete head on
the bottom page. This stage uses a DSPy ChainOfThought module to decide whether
each cross-page pair is a split block and merges the halves. Workers run in two
passes (even/odd) to avoid races on shared segments.

TWO QUESTIONS, TWO CALLS. The judge (``Signature``) answers one bool: are
these two halves of one block? Only if it says yes does
the rewriter (``MergeSignature``) get asked the second question — what is that
block — and rejoin them. Splitting them this way is not tidiness. A single
``merged: str | None`` field used to carry both answers, and asked to return
None for "these don't merge" the model returned the four-character STRING
"None", which is not None and is truthy: every declined seam overwrote the
tail with the word "None" and deleted the head, destroying two nodes at 100%
of page boundaries in every run, on two different models. Text is all a model
can emit, so a text field cannot hold "no" — the negative answer needs a field
whose type can, and here it needs a different call entirely.

The rewriter sees exactly what the judge saw — both edge nodes AND both
context neighbours — because where the interrupted block starts and stops is
the same question in both calls, and the neighbours are what settle it. The
context is read-only in both: it informs the answer, it is never part of it.

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

from kms.core import logs, models, state
from kms.core.recorder import Recorder

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

    Your job is ONE yes/no judgment: are the tail node of the top run and the
    head node of the bottom run two halves of the same interrupted block? You
    do not rewrite or join anything — a separate pass rejoins the halves, and
    only when you answer True.

    Judge this purely on structure: does the tail read as cut off mid-block and
    the head as its continuation? Do not reason about the subject matter or
    reassemble blocks that are each already complete. If the two are already
    complete, independent nodes that merely sit next to each other at the
    boundary, answer False.

    Use the context nodes (the neighbour just inside each run) only to inform
    your judgment — they are never part of the join.
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

    is_split: bool = dspy.OutputField(
        description='True if the tail node is cut off mid-block and the head node continues it, so the two are halves of one block. False if each is already complete on its own.'
    )


class MergeSignature(dspy.Signature):
    r"""
    Two texts are the two halves of ONE block of a document that a page break
    interrupted — the first is cut off, the second continues it. Write them
    back as the single block they were.

    You are here because this needs judgment. The break can fall anywhere —
    mid-word, mid-equation, mid-table-row, between two halves of a code fence —
    and no fixed rule joins all of those. Read what the two halves ARE and make
    them one coherent block again.

    ONE LINE YOU DO NOT CROSS: the content is the author's, not yours. Every
    word, number, symbol and LaTeX token of both halves survives, in its
    original order. You never paraphrase, summarise, translate, correct,
    reflow, drop a repeated word, or invent a word that was not in front of
    you. Within that, the SHAPE of the block is yours to repair.

    The ordinary cases:

      - A break INSIDE a word closes up: 'espe' + 'cially' is 'especially'.
      - A word broken across the break with a hyphen loses the hyphen:
        'sub-' + 'graph' is 'subgraph'.
      - A break BETWEEN words takes a single space: 'every vertex of' +
        '$G$ is a vertex' is 'every vertex of $G$ is a vertex'.
      - A structure split down the middle comes back as ONE structure. If the
        two halves are the two ends of a single display-math block, inline
        math, fenced code block, table or LaTeX environment, rejoin them into
        one well-formed whole — one opening delimiter, one closing delimiter,
        the two halves' content between them, in order.

    ONE STRUCTURE, NOT TWO STUCK TOGETHER. The page break often makes the
    front end close the structure at the foot of the page and reopen it at the
    top of the next, so BOTH halves arrive carrying their own delimiters. Drop
    the redundant pair in the middle — they exist only because of the break:

      - '$$x + y' + '= 4$$'          ->  '$$x + y = 4$$'
      - '$$x + y$$' + '$$= 4$$'      ->  '$$x + y = 4$$'
      - '$x +' + 'y$'                ->  '$x + y$'
      - '```python\nif p:' + 'return p\n```'
                                     ->  one fenced block, one pair of fences
      - '\begin{aligned} a &= b \\' + '\begin{aligned} c &= d \end{aligned}'
                                     ->  one aligned environment holding both
                                         rows

    Delimiters, fences and environment begin/end pairs that only mark where
    the page ended are yours to remove, add or move so the result is
    well-formed. Judge what the block needs. The content between them is not
    yours to touch.

    MARKUP IS NOT CONTENT, AND CONTENT IS NOT MARKUP. Repairing the structure
    does not license rewriting the author's notation. Leave the markup style
    exactly as you found it: `\(` stays `\(` and never becomes `\\(`; '$$'
    stays '$$' and never becomes '\['; escaping is neither added nor removed.
    You are closing a wound in the block, not restyling it.

    Do not comment on what you did. Return the rejoined block and nothing else.

    Use the context nodes (the neighbour just inside each page) only to tell
    where the interrupted block starts and stops — never include their content
    in what you return.
    """

    # PLAIN STRINGS, NOT SeamNodeDTO. A pydantic input field is rendered into
    # the prompt as JSON, and JSON escapes every backslash: a tail holding
    # `\(2^{p}\)` reaches the model as `\\(2^{p}\\)`. This module's whole
    # contract is reproducing characters exactly, and it was being shown
    # characters that were not the document's — it then copied the escaped
    # form back about a tenth of the time, silently doubling backslashes in
    # the healed text. No prompt wording can fix that; the field type can.
    tail: str = dspy.InputField(
        description='The first half — the block as it was cut off at the foot of the page.'
    )
    head: str = dspy.InputField(
        description='The second half — the block as it resumes at the top of the next page.'
    )
    tail_kind: str = dspy.InputField(
        description="The first half's structural kind (paragraph, math, list, code, table, …)."
    )
    head_kind: str = dspy.InputField(
        description="The second half's structural kind."
    )
    before_tail: str = dspy.InputField(
        description='The block before the tail on its page. Read-only context — never include it in the output. Empty if there is none.'
    )
    after_head: str = dspy.InputField(
        description='The block after the head on its page. Read-only context — never include it in the output. Empty if there is none.'
    )

    merged: str = dspy.OutputField(
        description='The two halves rejoined into one block, both preserved in full.'
    )


class SeamMerger(dspy.Module):
    """Decides whether two adjacent edge nodes are halves of one block.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.merger = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> bool:
        """Judge one seam.

        Args:
            top_bottom_edge_node: The tail node of the top run.
            bottom_top_edge_node: The head node of the bottom run.
            top_node_context: The neighbour just inside the top run.
            bottom_node_context: The neighbour just inside the bottom run.

        Returns:
            Whether the pair is the two halves of one interrupted block.
        """
        result = await self.merger.acall(
            top_node_context=top_node_context,
            top_bottom_edge_node=top_bottom_edge_node,
            bottom_top_edge_node=bottom_top_edge_node,
            bottom_node_context=bottom_node_context,
        )
        if self._recorder:
            self._recorder.record(
                'seam_merger',
                {
                    'top_bottom_edge_node': top_bottom_edge_node,
                    'bottom_top_edge_node': bottom_top_edge_node,
                    'top_node_context': top_node_context,
                    'bottom_node_context': bottom_node_context,
                },
                result,
            )
        return bool(result.is_split)

    def forward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> bool:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                top_bottom_edge_node=top_bottom_edge_node,
                bottom_top_edge_node=bottom_top_edge_node,
                top_node_context=top_node_context,
                bottom_node_context=bottom_node_context,
            )
        )


class SeamRewriter(dspy.Module):
    """Rejoins the two halves of a block the seam merger judged to be split.

    Its own module rather than a second predictor on ``SeamMerger``: this is
    the second QUESTION (what is the block?), asked only when the first was
    answered yes, and every question in this pipeline gets a module of its own
    so it can be stubbed, recorded and optimised separately. It stays in this
    file because it is meaningless away from the seam — the two questions are
    about the same pair of nodes, share ``SeamNodeDTO``, and are explained by
    the same module docstring.

    It is called inline by ``_merge_pair`` rather than run as its own graph
    stage. The even pass's merges are already written back before the odd pass
    dispatches (see the parity note below), so deferring the rejoin to a later
    stage would leave the odd pass judging against half-healed pages.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.rewriter = dspy.ChainOfThought(MergeSignature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> str:
        """Rejoin one seam's two halves.

        Takes the same four nodes the judge saw, context included: where the
        interrupted block starts and stops is the same question here as there,
        and the neighbours are what settle it.

        Args:
            top_bottom_edge_node: The first half, cut off at the foot of the
                page.
            bottom_top_edge_node: The second half, resuming on the next page.
            top_node_context: The neighbour just inside the top run.
            bottom_node_context: The neighbour just inside the bottom run.

        Returns:
            The rejoined block.
        """
        # Unpacked to plain strings on the way in — see the note on the
        # signature's fields.
        inputs = {
            'tail': top_bottom_edge_node.content or '',
            'head': bottom_top_edge_node.content or '',
            'tail_kind': ' '.join(top_bottom_edge_node.types),
            'head_kind': ' '.join(bottom_top_edge_node.types),
            'before_tail': (
                top_node_context.content if top_node_context else ''
            )
            or '',
            'after_head': (
                bottom_node_context.content if bottom_node_context else ''
            )
            or '',
        }
        result = await self.rewriter.acall(**inputs)
        if self._recorder:
            self._recorder.record('seam_rewriter', inputs, result)
        return result.merged

    def forward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> str:
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
    module: SeamMerger,
    rewriter: SeamRewriter,
    top: models.Segment,
    bottom: models.Segment,
) -> list[tuple[int, list[models.ASTNode]]]:
    """Merge one seam, if the LLM judges it to be a split node.

    The edges are the top's last and the bottom's first *mergeable* nodes —
    page apparatus is passed over on both sides, so a page that ends in a
    footnote still has its real tail healed. A healed seam folds the rejoined
    content into that tail and drops that head.

    Args:
        module: The seam-judging module.
        rewriter: The module that rejoins a split pair. Only called for a
            seam the judge accepts.
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

    edges = {
        'top_bottom_edge_node': _to_seam_node_dto(tail),
        'bottom_top_edge_node': _to_seam_node_dto(head),
        'top_node_context': _to_seam_node_dto(top_context),
        'bottom_node_context': _to_seam_node_dto(bottom_context),
    }
    is_split = await module.aforward(**edges)
    logger.debug(
        'seam %d/%d: %s | tail %r + head %r',
        top.index,
        bottom.index,
        'merged' if is_split else 'left split',
        logs.elide(tail.content, 40),
        logs.elide(head.content, 40),
    )
    if is_split:
        # The rewriter sees exactly what the judge saw, context included, and
        # writes the two halves back as the one block they were.
        tail.content = await rewriter.aforward(**edges)
        del bottom_nodes[head_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class SeamMergerNode:
    """Heals cross-page splits using two parity passes to avoid races.

    Args:
        module: The seam-judging module.
        rewriter: The module that rejoins a split pair.
    """

    def __init__(
        self,
        module: SeamMerger,
        rewriter: SeamRewriter,
    ) -> None:
        self.module = module
        self.rewriter = rewriter

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
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_even_results': merged}

    async def odd_worker(self, state: dict) -> dict:
        """Merges one odd pair and returns the healed segment nodes."""
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_odd_results': merged}

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
