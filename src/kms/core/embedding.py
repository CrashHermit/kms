"""Provider-neutral multimodal embedding interfaces."""

from dataclasses import dataclass
from typing import Literal, Protocol

from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Identity and vector dimension of the selected embedding model."""

    model: str
    dimension: int


class Embedder(Protocol):
    """Embed ordered multimodal materials using one explicit input mode."""

    async def embed(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: Literal['query', 'document'],
    ) -> list[list[float]]:
        """Return one vector per material in the same order."""
        ...
