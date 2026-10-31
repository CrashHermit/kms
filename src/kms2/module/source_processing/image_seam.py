"""DSPy module for indexed native-image seam judgments."""

import dspy

from kms2.core.model.source_processing.image_seam import (
    ImageSeamDecision,
    ImageSeamInput,
)


class ImageSeamSignature(dspy.Signature):
    """Judge each independent top/bottom pair for one interrupted visual.

    Each item addresses two images in seam_images using zero-based native-image
    positions. True means the bottom of the top image continues into the top of
    the bottom image as one visual object split by a page boundary. False means
    separate figures, unrelated images, or uncertain continuation. Use only the
    two addressed images; other items and pairs provide no reference context.

    Return one decision per item, preserving its outer position. Do not describe,
    transcribe, rewrite, stitch, or modify images, or supply stitching instructions.
    """

    items: list[ImageSeamInput] = dspy.InputField(
        description='Independent pairs of zero-based top/bottom native-image references.'
    )
    seam_images: list[dspy.Image] = dspy.InputField(
        description='Native images addressed by each pair in items.'
    )
    decisions: list[ImageSeamDecision] = dspy.OutputField(
        description='Indexed continuation judgments, one per pair.'
    )


class ImageSeamJudgeModule(dspy.Module):
    """Judge a packet of independent visual seam pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[ImageSeamInput], seam_images: list[dspy.Image]
    ) -> list[ImageSeamDecision]:
        """Judge a visual packet synchronously."""
        return self.predictor(items=items, seam_images=seam_images).decisions

    async def aforward(
        self, *, items: list[ImageSeamInput], seam_images: list[dspy.Image]
    ) -> list[ImageSeamDecision]:
        """Judge a visual packet asynchronously."""
        prediction = await self.predictor.acall(
            items=items, seam_images=seam_images
        )
        return prediction.decisions
