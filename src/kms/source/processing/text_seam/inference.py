"""DSPy modules for indexed text seam judgment and rewriting."""

import dspy

from kms.source.processing.text_seam.models import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteOutput,
)


class TextSeamSignature(dspy.Signature):
    """Judge each independent adjacent-block seam structurally.

    A page boundary can interrupt a paragraph, sentence, equation, list item,
    caption, or fenced structure. is_split is True only when the tail and head
    are two halves of that same interrupted block: the tail is cut off and the
    head continues it. Two already complete independent blocks are False, even
    when they discuss the same subject. Do not reconstruct thematic arguments.

    top_context and bottom_context are reference-only neighbors, never part of
    a join; None means no neighbor. Each item is independent. Do not use another
    item's text or context. Return one decision per item preserving its position.
    Do not rewrite, join, or answer anything.
    """

    items: list[TextSeamInput] = dspy.InputField(
        description='Independent tail/head candidates with structural kinds and reference neighbors.'
    )
    decisions: list[TextSeamDecision] = dspy.OutputField(
        description='Indexed judgments: True only for one interrupted block.'
    )


class TextSeamJudgeModule(dspy.Module):
    """Judge indexed candidate seams in one packet."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, items: list[TextSeamInput]) -> list[TextSeamDecision]:
        """Judge a seam packet synchronously."""
        return self.predictor(items=items).decisions

    async def aforward(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamDecision]:
        """Judge a seam packet asynchronously."""
        prediction = await self.predictor.acall(items=items)
        return prediction.decisions


class TextSeamRewriteSignature(dspy.Signature):
    r"""Rejoin each item's tail and head as ONE interrupted authored block.

    All supplied items have a positive split judgment. Treat each independently;
    other items are not evidence. Preserve its position exactly, including gaps
    in the positions retained from the original judgment packet.

    Preserve every authored word, number, symbol and LaTeX token in original
    order. Never paraphrase, summarize, translate, correct content, drop repeated
    authored words, or invent text. Repair only seam-induced shape: close a
    word split internally, remove a page-break hyphen inside a word, and insert
    appropriate whitespace between words. Keep indentation needed by structures.

    A split display, inline math, code fence, table or LaTeX environment becomes
    one well-formed structure. Remove redundant closing/reopening delimiters at
    the seam, retain one enclosing pair, and keep both halves' content in order.
    Delimiters/fences/environment boundaries caused by the page break may be
    added, removed or moved; their content may not be rewritten. Do not restyle
    notation, change delimiter conventions, or add/remove escaping.

    top_context/bottom_context only locate the block boundaries; never include
    their content in merged. None means no reference neighbor. Return one
    indexed rewrite per item, with only the rejoined block in merged and no
    commentary, field headings or transport markers.
    """

    items: list[TextSeamInput] = dspy.InputField(
        description='Positive seam items with original positions, typed halves and reference neighbors.'
    )
    rewrites: list[TextSeamRewriteOutput] = dspy.OutputField(
        description='Original position and authored halves rejoined as one block.'
    )


class TextSeamRewriterModule(dspy.Module):
    """Rejoin a packet of positively judged text seams."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamRewriteOutput]:
        """Rewrite a positive packet synchronously."""
        return self.predictor(items=items).rewrites

    async def aforward(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamRewriteOutput]:
        """Rewrite a positive packet asynchronously."""
        prediction = await self.predictor.acall(items=items)
        return prediction.rewrites
