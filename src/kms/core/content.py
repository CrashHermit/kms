"""Project readable material into one request-local native-image catalog."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kms.core.model.content import MultimodalContent


@dataclass(frozen=True, slots=True)
class ContentImage:
    """An immutable image identified independently of its evidence provenance."""

    key: tuple[str, ...]
    payload: Path | bytes


class MultimodalContentBuilder:
    """Build ordered content leaves sharing one prediction's image catalog."""

    def __init__(self) -> None:
        self.images: list[ContentImage] = []
        self._positions: dict[tuple[str, ...], int] = {}

    def build(
        self,
        *,
        text: str | None,
        images: Sequence[ContentImage] = (),
    ) -> MultimodalContent:
        positions: list[int] = []
        for image in images:
            position = self._positions.get(image.key)
            if position is None:
                position = len(self.images)
                self._positions[image.key] = position
                self.images.append(image)
            if position not in positions:
                positions.append(position)
        return MultimodalContent(text=text, image_positions=positions)
