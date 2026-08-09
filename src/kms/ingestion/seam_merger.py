import asyncio
import logging

import dspy
from langgraph.types import Send
from pydantic import BaseModel

from kms.core import logs, models, recording, state

logger = logging.getLogger(__name__)


class SeamNodeDTO(BaseModel):
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
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
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
        return result.is_split

    def forward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> bool:
        return asyncio.run(
            self.aforward(
                top_bottom_edge_node=top_bottom_edge_node,
                bottom_top_edge_node=bottom_top_edge_node,
                top_node_context=top_node_context,
                bottom_node_context=bottom_node_context,
            )
        )


class SeamRewriter(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
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
        return asyncio.run(
            self.aforward(
                top_bottom_edge_node=top_bottom_edge_node,
                bottom_top_edge_node=bottom_top_edge_node,
                top_node_context=top_node_context,
                bottom_node_context=bottom_node_context,
            )
        )


def _to_seam_node_dto(node: models.ASTNode | None) -> SeamNodeDTO:
    if node is None:
        return SeamNodeDTO(content=None, types=[])
    return SeamNodeDTO(content=node.content, types=[node.type])

_APPARATUS = {'bibliographic', 'note'}


def _mergeable_indices(nodes: list[models.ASTNode]) -> list[int]:
    return [
        index for index, node in enumerate(nodes) if node.type not in _APPARATUS
    ]


def _pairs(
    segments: list[models.Segment], parity: int
) -> list[tuple[models.Segment, models.Segment]]:
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
        tail.content = await rewriter.aforward(**edges)
        del bottom_nodes[head_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class SeamMergerNode:
    def __init__(
        self,
        module: SeamMerger,
        rewriter: SeamRewriter,
    ) -> None:
        self.module = module
        self.rewriter = rewriter

    def dispatch_even(self, state: state.State) -> list[Send] | str:
        pairs = _pairs(state.get('segments', []), parity=0)
        sends = [
            Send('seam_even_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_even_collect'

    def dispatch_odd(self, state: state.State) -> list[Send] | str:
        pairs = _pairs(state.get('segments', []), parity=1)
        sends = [
            Send('seam_odd_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'seam_odd_collect'

    async def even_worker(self, state: dict) -> dict:
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_even_results': merged}

    async def odd_worker(self, state: dict) -> dict:
        merged = await _merge_pair(
            self.module, self.rewriter, state['top'], state['bottom']
        )
        return {'seam_odd_results': merged}

    def _collect(self, state: state.State, channel: str) -> dict:
        segments = models.merge_results_into_segments(
            state['segments'], state.get(channel, []), 'nodes'
        )
        return {'segments': segments}

    def even_collect(self, state: state.State) -> dict:
        return self._collect(state, 'seam_even_results')

    def odd_collect(self, state: state.State) -> dict:
        result = self._collect(state, 'seam_odd_results')
        segments = result['segments']
        nodes = models.flatten_segments(segments)
        logger.info(
            'seam merger: %d page(s) -> flat stream of %d node(s)',
            len(segments),
            len(nodes),
        )
        return {
            'segments': segments,
            'nodes': nodes,
        }

