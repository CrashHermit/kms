"""Detects and rejoins blocks split across segment (page) seams."""

import hashlib
import logging
from pathlib import Path

import dspy
from langgraph.types import Send
from PIL import Image

from kms.core import content, logs, models, module, state, walker

logger = logging.getLogger(__name__)


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

    top_node_context: content.ContentParts = dspy.InputField(
        description='The node immediately before the tail, as multimodal read-only context.'
    )
    top_bottom_edge_node: content.ContentParts = dspy.InputField(
        description='The tail node of the top element run, including text and image content.'
    )
    bottom_top_edge_node: content.ContentParts = dspy.InputField(
        description='The head node of the bottom element run, including text and image content.'
    )
    bottom_node_context: content.ContentParts = dspy.InputField(
        description='The node immediately after the head, as multimodal read-only context.'
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

    tail: content.ContentParts = dspy.InputField(
        description='The first half — the block as it was cut off at the foot of the page.'
    )
    head: content.ContentParts = dspy.InputField(
        description='The second half — the block as it resumes at the top of the next page.'
    )
    tail_kind: str = dspy.InputField(
        description="The first half's structural kind (paragraph, math, list, code, table, …)."
    )
    head_kind: str = dspy.InputField(
        description="The second half's structural kind."
    )
    before_tail: content.ContentParts = dspy.InputField(
        description='The block before the tail on its page, as read-only multimodal context.'
    )
    after_head: content.ContentParts = dspy.InputField(
        description='The block after the head on its page, as read-only multimodal context.'
    )

    merged: str = dspy.OutputField(
        description='The two halves rejoined into one block, both preserved in full.'
    )


class SeamMerger(module.Module):
    """Decides whether two edge nodes are halves of one split block."""

    signature = Signature
    record_name = 'seam_merger'

    def encode(
        self,
        top_bottom_edge_node: walker.WindowNode,
        bottom_top_edge_node: walker.WindowNode,
        top_node_context: walker.WindowNode | None = None,
        bottom_node_context: walker.WindowNode | None = None,
    ) -> dict:
        """Builds multimodal seam-signature kwargs for one edge pair."""
        return {
            'top_node_context': _content_parts(top_node_context),
            'top_bottom_edge_node': _content_parts(top_bottom_edge_node),
            'bottom_top_edge_node': _content_parts(bottom_top_edge_node),
            'bottom_node_context': _content_parts(bottom_node_context),
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns whether the edge nodes are one interrupted block."""
        return module.require_bool(prediction.is_split, 'is_split')


class SeamRewriter(module.Module):
    """Rejoins two halves of a split block into one coherent block."""

    signature = MergeSignature
    record_name = 'seam_rewriter'

    def encode(
        self,
        top_bottom_edge_node: walker.WindowNode,
        bottom_top_edge_node: walker.WindowNode,
        top_node_context: walker.WindowNode | None = None,
        bottom_node_context: walker.WindowNode | None = None,
    ) -> dict:
        """Builds multimodal merge-signature kwargs for one edge pair."""
        return {
            'tail': _content_parts(top_bottom_edge_node),
            'head': _content_parts(bottom_top_edge_node),
            'tail_kind': top_bottom_edge_node.type or '',
            'head_kind': bottom_top_edge_node.type or '',
            'before_tail': _content_parts(top_node_context),
            'after_head': _content_parts(bottom_node_context),
        }

    def decode(self, prediction, **inputs) -> str:
        """Returns the rejoined block text for the two halves."""
        return module.require_text(prediction.merged, 'merged')


def _to_window_node(node: models.Node | None) -> walker.WindowNode:
    """Projects a seam node into the shared multimodal window view."""
    if node is None:
        return walker.WindowNode(position=0)
    return walker.WindowNode(
        position=0,
        type=node.type,
        content=node.content,
        image_path=node.image_path,
    )


def _content_parts(node: walker.WindowNode | None) -> content.ContentParts:
    """Converts one seam view into the canonical DSPy content boundary."""
    if node is None:
        return content.ContentParts(content=content.Content())
    return content.labeled_content_parts([node])


def _image_only_node(node: models.Node) -> bool:
    """Returns whether an image node has no independent textual payload."""
    return bool(
        node.image_path
        and node.type == models.NodeType.IMAGE
        and (
            not node.content
            or node.content.lstrip().startswith('![')
        )
    )


def _merge_images(top_path: str, bottom_path: str) -> str:
    """Stitches two page-seam image assets into one deterministic PNG."""
    top = Path(top_path)
    bottom = Path(bottom_path)
    digest = hashlib.sha256(
        top.read_bytes() + b'\\0' + bottom.read_bytes()
    ).hexdigest()[:16]
    output = top.with_name(f'{top.stem}--seam-{digest}.png')
    if output.exists():
        return str(output)
    with Image.open(top) as first, Image.open(bottom) as second:
        images = [first.convert('RGBA'), second.convert('RGBA')]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        canvas = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        y = 0
        for image in images:
            canvas.paste(image, ((width - image.width) // 2, y), image)
            y += image.height
        canvas.save(output, format='PNG')
    return str(output)


_APPARATUS = {'bibliographic', 'note'}


def _mergeable_indices(nodes: list[models.Node]) -> list[int]:
    """Returns the indices of nodes that can take part in a seam merge."""
    return [
        index for index, node in enumerate(nodes) if node.type not in _APPARATUS
    ]


def _pairs(
    documents: list[models.Document], parity: int
) -> list[tuple[models.Document, models.Document]]:
    """Returns adjacent segment pairs of the given index parity."""
    return [
        (documents[i], documents[i + 1])
        for i in range(len(documents) - 1)
        if documents[i].index % 2 == parity
        and _mergeable_indices(documents[i].nodes)
        and _mergeable_indices(documents[i + 1].nodes)
    ]


async def _merge_pair(
    module: SeamMerger,
    rewriter: SeamRewriter,
    top: models.Document,
    bottom: models.Document,
) -> list[tuple[int, list[models.Node]]]:
    """Judges and merges one segment pair, returning both node lists.

    When the seam is judged split, the tail is rewritten in place with
    the joined block and the bottom's head node is deleted.
    """
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)

    top_mergeable = _mergeable_indices(top_nodes)
    bottom_mergeable = _mergeable_indices(bottom_nodes)
    if not top_mergeable or not bottom_mergeable:
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
        'top_bottom_edge_node': _to_window_node(tail),
        'bottom_top_edge_node': _to_window_node(head),
        'top_node_context': _to_window_node(top_context),
        'bottom_node_context': _to_window_node(bottom_context),
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
        if _image_only_node(tail) and _image_only_node(head):
            original_image_paths = [tail.image_path, head.image_path]
            tail.image_path = _merge_images(
                tail.image_path, head.image_path
            )
            tail.content = None
            tail.provenance = {
                **tail.provenance,
                'seam_merged_from': original_image_paths,
            }
        elif tail.content or head.content:
            tail.content = await rewriter.aforward(**edges)
        else:
            logger.warning(
                'seam %d/%d judged split but has no mergeable text or image pair',
                top.index,
                bottom.index,
            )
            return [(top.index, top_nodes), (bottom.index, bottom_nodes)]
        del bottom_nodes[head_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class SeamMergerNode:
    """Two-phase langgraph node merging seams on even/odd pairs.

    Even and odd pairs are processed in separate passes so that
    renumbered parities from the previous pass stay independent.
    """

    def __init__(
        self,
        module: SeamMerger,
        rewriter: SeamRewriter,
    ) -> None:
        self.module = module
        self.rewriter = rewriter

    def dispatch_even(self, state: state.State) -> list[Send] | str:
        """Sends one worker per even-indexed adjacent segment pair."""
        pairs = _pairs(state.get('documents', []), parity=0)
        sends = [
            Send('seam_even_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_even_collect'

    def dispatch_odd(self, state: state.State) -> list[Send] | str:
        """Sends one worker per odd-indexed adjacent segment pair."""
        pairs = _pairs(state.get('documents', []), parity=1)
        sends = [
            Send('seam_odd_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_odd_collect'

    async def even_worker(self, state: dict) -> dict:
        """Merges one even-indexed pair and reports its node lists."""
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_even_results': merged}

    async def odd_worker(self, state: dict) -> dict:
        """Merges one odd-indexed pair and reports its node lists."""
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_odd_results': merged}

    def _collect(self, state: state.State, channel: str) -> dict:
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
