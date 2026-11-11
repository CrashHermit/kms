"""DSPy module for contextual image retention decisions."""

import dspy

from kms.source.processing.image_filter.models import (
    ImageFilterDecision,
    ImageFilterInput,
)


class ImageFilterSignature(dspy.Signature):
    """Judge each target image using only that target's assigned context.

    Multiple target assets are ordered parts of one logical image. Context is
    read-only and may contain text or neighboring images. Keep diagrams, charts,
    photographs relevant to the source, equation images, screenshots,
    substantive illustrations, and uncertain images. Remove clear website
    decoration, unrelated logos or icons, advertisements, and interface
    artifacts. Retain an advertisement when the surrounding text studies it as
    instructional subject matter. Do not filter by a particular teaching goal,
    judge truth, describe or transcribe images, edit pixels, or stitch images.
    A decision applies only to its target, never to neighboring blocks, and
    must not reject a useful mixed figure because it contains a small artifact.

    Return exactly one Boolean decision per input, preserving each zero-based
    position.
    """

    items: list[ImageFilterInput] = dspy.InputField(
        description='Independent image targets with their own source context.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native assets referenced by targets and context.'
    )
    decisions: list[ImageFilterDecision] = dspy.OutputField(
        description='One keep-or-remove decision for each indexed target.'
    )


class ImageFilterModule(dspy.Module):
    """Return contextual keep decisions for logical images."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[ImageFilterInput]
    ) -> list[ImageFilterDecision]:
        """Filter one packet synchronously."""
        return self.predictor(items=items).decisions

    async def aforward(
        self, *, items: list[ImageFilterInput]
    ) -> list[ImageFilterDecision]:
        """Filter one packet asynchronously."""
        prediction = await self.predictor.acall(items=items)
        return prediction.decisions
