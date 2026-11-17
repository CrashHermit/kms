"""Shared model-drafted native learning materials."""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.media.models import ImageAsset, ImagePart, Material, TextPart


class ImageReference(BaseModel):
    """Reference an image from the request's native image catalog."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['image'] = 'image'
    image_position: int


class MaterialDraft(BaseModel):
    """Ordered material with model-local native-image references."""

    model_config = ConfigDict(extra='forbid')

    parts: tuple[
        Annotated[TextPart | ImageReference, Field(discriminator='type')], ...
    ]


def resolve_material(
    draft: MaterialDraft, image_assets: Sequence[ImageAsset]
) -> Material:
    """Resolve model-local image indexes to supplied request assets."""
    return Material(
        parts=tuple(
            ImagePart(asset=image_assets[part.image_position])
            if part.type == 'image'
            else TextPart(text=part.text)
            for part in draft.parts
        )
    )
