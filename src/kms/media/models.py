"""Shared ordered material and native image models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class ImageAsset:
    """An immutable native image identified by a stable key."""

    key: tuple[str, ...]
    payload: Path | bytes


class TextPart(BaseModel):
    """One text occurrence in ordered material."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    type: Literal['text'] = 'text'
    text: str


class ImagePart(BaseModel):
    """One native-image occurrence in ordered material."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    type: Literal['image'] = 'image'
    asset: ImageAsset


MaterialPart = Annotated[TextPart | ImagePart, Field(discriminator='type')]


class Material(BaseModel):
    """Immutable text and native-image parts in encounter order."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    parts: tuple[MaterialPart, ...] = ()

    @classmethod
    def from_text(cls, text: str | None) -> Material:
        """Create material containing exactly the supplied text, if present."""
        return cls(parts=() if text is None else (TextPart(text=text),))

    @classmethod
    def from_image(cls, asset: ImageAsset) -> Material:
        """Create material containing one native-image occurrence."""
        return cls(parts=(ImagePart(asset=asset),))

    @property
    def text(self) -> str | None:
        """Concatenate text occurrences, or return None when there are none."""
        texts = [part.text for part in self.parts if isinstance(part, TextPart)]
        return ''.join(texts) if texts else None

    @property
    def images(self) -> tuple[ImageAsset, ...]:
        """Return native-image occurrences in encounter order."""
        return tuple(
            part.asset for part in self.parts if isinstance(part, ImagePart)
        )
