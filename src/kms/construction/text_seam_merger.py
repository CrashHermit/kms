"""Detects and rejoins blocks split across segment (page) seams."""

import logging
from typing import Any

import dspy
from langgraph.types import Send

from kms.core import logs, models, module, state

logger = logging.getLogger(__name__)


class TextSeamSignature(dspy.Signature):
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

    top_node_context: str = dspy.InputField(
        description='Text from the node immediately before the tail, if available.'
    )
    top_bottom_edge_node: str = dspy.InputField(
        description='The complete text of the tail source node.'
    )
    bottom_top_edge_node: str = dspy.InputField(
        description='The complete text of the head source node.'
    )
    bottom_node_context: str = dspy.InputField(
        description='Text from the node immediately after the head, if available.'
    )

    is_split: bool = dspy.OutputField(
        description='True if the tail node is cut off mid-block and the head node continues it, so the two are halves of one block. False if each is already complete on its own.'
    )


class TextSeamRewriteSignature(dspy.Signature):
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

    tail: str = dspy.InputField(
        description='The first text half, cut off at the page boundary.'
    )
    head: str = dspy.InputField(
        description='The second text half, resuming after the page boundary.'
    )
    tail_kind: str = dspy.InputField(
        description="The first half's structural kind."
    )
    head_kind: str = dspy.InputField(
        description="The second half's structural kind."
    )
    before_tail: str = dspy.InputField(
        description='Text from the block before the tail, if available.'
    )
    after_head: str = dspy.InputField(
        description='Text from the block after the head, if available.'
    )

    merged: str = dspy.OutputField(
        description='The two halves rejoined into one block, both preserved in full.'
    )


class TextSeamMerger(module.Module):
    """Decides whether two edge nodes are halves of one split block."""

    signature = TextSeamSignature
    record_name = 'text_seam_merger'

    def encode(
        self,
        top_bottom_edge_node: models.SourceNode,
        bottom_top_edge_node: models.SourceNode,
        top_node_context: models.SourceNode | None = None,
        bottom_node_context: models.SourceNode | None = None,
    ) -> dict[str, str]:
        """Builds text-only seam-signature kwargs for one edge pair."""
        return {
            'top_node_context': _text(top_node_context),
            'top_bottom_edge_node': _text(top_bottom_edge_node),
            'bottom_top_edge_node': _text(bottom_top_edge_node),
            'bottom_node_context': _text(bottom_node_context),
        }

    def decode(self, prediction: Any, **inputs: Any) -> bool:
        """Returns whether the edge nodes are one interrupted block."""
        return module.require_bool(prediction.is_split, 'is_split')


class TextSeamRewriter(module.Module):
    """Rejoins two halves of a split block into one coherent block."""

    signature = TextSeamRewriteSignature
    record_name = 'text_seam_rewriter'

    def encode(
        self,
        top_bottom_edge_node: models.SourceNode,
        bottom_top_edge_node: models.SourceNode,
        top_node_context: models.SourceNode | None = None,
        bottom_node_context: models.SourceNode | None = None,
    ) -> dict[str, str]:
        """Builds text-only merge-signature kwargs for one edge pair."""
        return {
            'tail': _text(top_bottom_edge_node),
            'head': _text(bottom_top_edge_node),
            'tail_kind': top_bottom_edge_node.type or '',
            'head_kind': bottom_top_edge_node.type or '',
            'before_tail': _text(top_node_context),
            'after_head': _text(bottom_node_context),
        }

    def decode(self, prediction: Any, **inputs: Any) -> str:
        """Returns the rejoined block text for the two halves."""
        return module.require_text(prediction.merged, 'merged')


def _text(node: models.SourceNode | None) -> str:
    """Returns source text for the text-only DSPy boundary."""
    return node.content or '' if node is not None else ''


def _text_mergeable(node: models.SourceNode) -> bool:
    """Returns whether a node is safe for the text-only seam path."""
    return bool(
        node.content
        and node.type not in _APPARATUS
        and node.type != models.NodeType.IMAGE
        and not node.assets
    )


_APPARATUS = {'bibliographic', 'note'}


def _skip_for_text(node: models.SourceNode) -> bool:
    """Returns whether text seam selection should pass over this node."""
    return bool(
        node.type in _APPARATUS
        or node.type == models.NodeType.IMAGE
        or node.assets
    )


def _edge_index(nodes: list[models.SourceNode], *, reverse: bool) -> int | None:
    """Returns the nearest text edge, skipping apparatus and visual nodes."""
    indices = range(len(nodes) - 1, -1, -1) if reverse else range(len(nodes))
    for index in indices:
        node = nodes[index]
        if _skip_for_text(node):
            continue
        return index if _text_mergeable(node) else None
    return None


def _context_node(
    nodes: list[models.SourceNode], start: int, step: int
) -> models.SourceNode | None:
    """Finds nearby text context while skipping apparatus and visual nodes."""
    for index in range(start, len(nodes) if step > 0 else -1, step):
        node = nodes[index]
        if _skip_for_text(node):
            continue
        return node if _text_mergeable(node) else None
    return None


def _pairs(
    documents: list[models.Document], parity: int
) -> list[tuple[models.Document, models.Document]]:
    """Returns adjacent text-compatible segment pairs of the given parity."""
    return [
        (documents[i], documents[i + 1])
        for i in range(len(documents) - 1)
        if documents[i].index % 2 == parity
        and _edge_index(documents[i].nodes, reverse=True) is not None
        and _edge_index(documents[i + 1].nodes, reverse=False) is not None
    ]


async def _merge_pair(
    module: TextSeamMerger,
    rewriter: TextSeamRewriter,
    top: models.Document,
    bottom: models.Document,
) -> list[tuple[int, list[models.SourceNode]]]:
    """Judges and merges one text-compatible segment pair."""
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)

    tail_index = _edge_index(top_nodes, reverse=True)
    head_index = _edge_index(bottom_nodes, reverse=False)
    if tail_index is None or head_index is None:
        return [(top.index, top_nodes), (bottom.index, bottom_nodes)]

    tail = top_nodes[tail_index]
    head = bottom_nodes[head_index]
    top_context = _context_node(top_nodes, tail_index - 1, -1)
    bottom_context = _context_node(bottom_nodes, head_index + 1, 1)

    edges = {
        'top_bottom_edge_node': tail,
        'bottom_top_edge_node': head,
        'top_node_context': top_context,
        'bottom_node_context': bottom_context,
    }
    is_split = await module.aforward(**edges)
    logger.debug(
        'seam %d/%d: %s | tail %r + head %r',
        bottom.index,
        'merged' if is_split else 'left split',
        logs.elide(tail.content, 40),
        logs.elide(head.content, 40),
    )
    if is_split:
        tail.content = await rewriter.aforward(**edges)
        del bottom_nodes[head_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class TextSeamMergerNode:
    """Two-phase langgraph node merging seams on even/odd pairs.

    Even and odd pairs are processed in separate passes so that
    renumbered parities from the previous pass stay independent.
    """

    def __init__(
        self,
        module: TextSeamMerger,
        rewriter: TextSeamRewriter,
    ) -> None:
        self.module = module
        self.rewriter = rewriter

    def dispatch_even(self, state: state.State) -> list[Send] | str:
        """Sends one worker per even-indexed adjacent segment pair."""
        pairs = _pairs(state.get('documents', []), parity=0)
        sends = [
            Send('text_seam_even_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'text_seam_even_collect'

    def dispatch_odd(self, state: state.State) -> list[Send] | str:
        """Sends one worker per odd-indexed adjacent segment pair."""
        pairs = _pairs(state.get('documents', []), parity=1)
        sends = [
            Send('text_seam_odd_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'text_seam_odd_collect'

    async def even_worker(self, state: dict[str, Any]) -> dict[str, Any]:
        """Merges one even-indexed pair and reports its node lists."""
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_even_results': merged}

    async def odd_worker(self, state: dict[str, Any]) -> dict[str, Any]:
        """Merges one odd-indexed pair and reports its node lists."""
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_odd_results': merged}

    def _collect(self, state: dict[str, Any], channel: str) -> dict[str, Any]:
        """Merges one channel's worker results back onto documents."""
        documents = state['documents']
        by_index = dict(state.get(channel, []))
        for document in documents:
            if document.index in by_index:
                document.nodes = by_index[document.index]
        return {'documents': documents}

    def even_collect(self, state: state.State) -> dict:
        """Collects the even-pass results."""
        return self._collect(state, 'seam_even_results')

    def odd_collect(self, state: state.State) -> dict:
        """Collects the odd-pass results and flattens the final stream."""
        result = self._collect(state, 'seam_odd_results')
        documents = result['documents']
        nodes = models.flatten_documents(documents)
        logger.info(
            'seam merger: %d document(s) -> flat stream of %d node(s)',
            len(documents),
            len(nodes),
        )
        return {
            'documents': documents,
            'nodes': nodes,
        }
