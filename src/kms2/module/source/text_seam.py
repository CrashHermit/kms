"""DSPy modules for judging and rewriting text seams."""

import dspy

from kms2.core.model.source import SourceBlock


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


def _text(source_block: SourceBlock | None) -> str:
    return source_block.content or '' if source_block is not None else ''


class TextSeamJudgeModule(dspy.Module):
    """Make one boolean judgment about whether a page seam splits text."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(TextSeamSignature)
        self.predictor.set_lm(language_model)

    def _inputs(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None,
        bottom_context: SourceBlock | None,
    ) -> dict[str, str]:
        return {
            'top_node_context': _text(top_context),
            'top_bottom_edge_node': _text(tail),
            'bottom_top_edge_node': _text(head),
            'bottom_node_context': _text(bottom_context),
        }

    def forward(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None = None,
        bottom_context: SourceBlock | None = None,
    ) -> bool:
        """Judge one adjacent pair synchronously."""
        prediction = self.predictor(
            **self._inputs(
                tail=tail,
                head=head,
                top_context=top_context,
                bottom_context=bottom_context,
            )
        )
        return prediction.is_split

    async def aforward(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None = None,
        bottom_context: SourceBlock | None = None,
    ) -> bool:
        """Judge one adjacent pair asynchronously."""
        prediction = await self.predictor.acall(
            **self._inputs(
                tail=tail,
                head=head,
                top_context=top_context,
                bottom_context=bottom_context,
            )
        )
        return prediction.is_split


class TextSeamRewriterModule(dspy.Module):
    """Rejoin two text blocks after a positive seam judgment."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(TextSeamRewriteSignature)
        self.predictor.set_lm(language_model)

    def _inputs(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None,
        bottom_context: SourceBlock | None,
    ) -> dict[str, str]:
        return {
            'tail': _text(tail),
            'head': _text(head),
            'tail_kind': tail.block_type.value,
            'head_kind': head.block_type.value,
            'before_tail': _text(top_context),
            'after_head': _text(bottom_context),
        }

    def forward(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None = None,
        bottom_context: SourceBlock | None = None,
    ) -> str:
        """Rewrite one adjacent pair synchronously."""
        prediction = self.predictor(
            **self._inputs(
                tail=tail,
                head=head,
                top_context=top_context,
                bottom_context=bottom_context,
            )
        )
        return prediction.merged

    async def aforward(
        self,
        *,
        tail: SourceBlock,
        head: SourceBlock,
        top_context: SourceBlock | None = None,
        bottom_context: SourceBlock | None = None,
    ) -> str:
        """Rewrite one adjacent pair asynchronously."""
        prediction = await self.predictor.acall(
            **self._inputs(
                tail=tail,
                head=head,
                top_context=top_context,
                bottom_context=bottom_context,
            )
        )
        return prediction.merged
