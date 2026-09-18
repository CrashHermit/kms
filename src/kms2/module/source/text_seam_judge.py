"""DSPy module for judging text seams."""

import dspy

from kms2.core.model import SourceBlock


class TextSeamSignature(dspy.Signature):
    """
    You are an expert technical editor. Two adjacent runs of document blocks
    share a seam — the boundary where one run ends and the next begins.
    Sometimes a single block (paragraph, sentence, equation, list item, caption,
    etc.) is split across that boundary, producing an incomplete tail in the top
    run and an incomplete head in the bottom run.

    Your job is ONE yes/no judgment: are the tail node of the top run and the
    head node of the bottom run two halves of the same interrupted block? You do
    not rewrite or join anything — a separate pass rejoins the halves, and only
    when you answer True.

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


def _text(source_block: SourceBlock | None) -> str:
    return source_block.content or '' if source_block is not None else ''


class TextSeamJudgeModule(dspy.Module):
    """Make one boolean judgment about whether a page seam splits text."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

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


__all__ = ['TextSeamSignature', 'TextSeamJudgeModule']
