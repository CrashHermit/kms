"""DSPy module for indexed positive-only text seam rewrites."""

import dspy

from kms.core.model.source_processing.text_seam import (
    TextSeamInput,
    TextSeamRewriteOutput,
)


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
