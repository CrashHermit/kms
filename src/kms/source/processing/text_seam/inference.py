"""DSPy modules for indexed text seam judgment and ordered-chain rewriting."""

import dspy

from kms.source.processing.text_seam.models import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteInput,
    TextSeamRewriteOutput,
)


class TextSeamSignature(dspy.Signature):
    """Judge whether independent adjacent blocks are one interrupted block.

    Return true only when the authored block continues across the physical page
    boundary. Preserve separate paragraphs, headings, lists, tables, and other
    independently authored structures as separate blocks.
    """

    items: list[TextSeamInput] = dspy.InputField(
        description='Independent tail/head candidates with structural context.'
    )
    decisions: list[TextSeamDecision] = dspy.OutputField(
        description='Packet-local split decisions for the candidate seams.'
    )


class TextSeamJudgeModule(dspy.Module):
    """Judge indexed candidate seams in one packet."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, items: list[TextSeamInput]) -> list[TextSeamDecision]:
        return self.predictor(items=items).decisions

    async def aforward(
        self, *, items: list[TextSeamInput]
    ) -> list[TextSeamDecision]:
        prediction = await self.predictor.acall(items=items)
        return prediction.decisions


class TextSeamRewriteSignature(dspy.Signature):
    r"""Rejoin each ordered chain as ONE interrupted authored block.

    Every item has a positive seam judgment at every internal boundary. Treat
    each chain independently. Preserve every authored word, number, symbol and
    LaTeX token in original order. Never paraphrase, summarize, correct content,
    drop repeated authored words, or invent text. Repair only seam-induced
    shape, such as a page-break hyphen or whitespace. Keep structural
    delimiters well-formed without changing their content.

    Outer contexts locate the chain boundary and are never included. Return one
    indexed rewrite per input with only the joined authored block in merged.
    """

    items: list[TextSeamRewriteInput] = dspy.InputField(
        description='Positive ordered chains of interrupted source blocks.'
    )
    rewrites: list[TextSeamRewriteOutput] = dspy.OutputField(
        description='Original packet-local position and joined chain text.'
    )


class TextSeamRewriterModule(dspy.Module):
    """Rejoin a packet of positively judged ordered chains."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[TextSeamRewriteInput]
    ) -> list[TextSeamRewriteOutput]:
        return self.predictor(items=items).rewrites

    async def aforward(
        self, *, items: list[TextSeamRewriteInput]
    ) -> list[TextSeamRewriteOutput]:
        prediction = await self.predictor.acall(items=items)
        return prediction.rewrites
