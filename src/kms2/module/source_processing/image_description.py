"""DSPy module for indexed logical-image descriptions."""

import dspy

from kms2.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionOutput,
)


class ImageDescriptionSignature(dspy.Signature):
    """Describe every item as one independent logical image.

    image_positions select ordered assets from target_images. Multiple assets
    for one item are parts of that one logical image, in source order. Use only
    those assets and that item's context_before/context_after; other images and
    other items' context are not evidence. Context is reference only, not a
    request to solve nearby tasks. Describe visible features without inventing
    labels, values, geometry, or answers. Do not combine unrelated images.

    Preserve explicit variable labels, legends, and associated units. Describe
    graph branches, open versus filled markers, endpoints, and gaps. Keep
    unannotated point locations qualitative. Do not transcribe numeric axis
    ticks or derive coordinates, axis extents, or intervals from grid squares:
    the original asset retains that visual data for direct inspection.
    Numeric point coordinates may be copied only from an annotation explicitly
    identifying that point. Do not solve nearby tasks or promote requested
    plotting conditions into a curve that is not drawn.
    Use an item's own context to identify explicitly associated variables,
    units, captions, and panel labels. Do not transfer unrelated context facts
    into the image. If a panel's identity is not established by its assigned
    assets or own context, do not invent a panel/function association.

    Return one description per item, preserving its zero-based position.
    Description strings contain only source-grounded prose and notation, never
    transport markers, field headings, or completion delimiters.
    """

    items: list[ImageDescriptionInput] = dspy.InputField(
        description='Independent logical images with ordered asset references and own context.'
    )
    target_images: list[dspy.Image] = dspy.InputField(
        description='Native visual assets addressed by each item, zero-based.'
    )
    descriptions: list[ImageDescriptionOutput] = dspy.OutputField(
        description='One indexed source-grounded description for each logical image.'
    )


class ImageDescriptionModule(dspy.Module):
    """Describe independently contextualized logical images in a packet."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[ImageDescriptionInput],
        target_images: list[dspy.Image],
    ) -> list[ImageDescriptionOutput]:
        """Describe a packet synchronously."""
        return self.predictor(
            items=items, target_images=target_images
        ).descriptions

    async def aforward(
        self,
        *,
        items: list[ImageDescriptionInput],
        target_images: list[dspy.Image],
    ) -> list[ImageDescriptionOutput]:
        """Describe a packet asynchronously."""
        prediction = await self.predictor.acall(
            items=items, target_images=target_images
        )
        return prediction.descriptions
