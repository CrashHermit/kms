"""DSPy module for indexed text seam judgments."""

import dspy

from kms2.core.model.source_processing.text_seam import (
    TextSeamDecision,
    TextSeamInput,
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
