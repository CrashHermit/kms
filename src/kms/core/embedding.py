"""Provider-neutral multimodal embedding interfaces."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from kms.media.models import Material


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Identity and vector dimension of the selected embedding model."""

    model: str
    dimension: int


class EmbeddingRequest(BaseModel):
    """Ordered materials and their embedding input mode."""

    model_config = ConfigDict(extra='forbid')

    materials: list[Material]
    input_type: Literal['query', 'document']


class Embedder(Protocol):
    """Embed ordered multimodal materials using one explicit input mode."""

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        """Return one vector per material in the same order."""
        ...


class EmbeddingEstimator(Protocol):
    """Measure model-specific input costs in request order."""

    def count_inputs(self, request: EmbeddingRequest) -> list[int]:
        """Return one estimated token cost per material."""
        ...


class EmbeddingCapacity(Protocol):
    """Enforce atomic inputs and determine whether whole-input batches fit."""

    def validate_items(
        self, request: EmbeddingRequest, costs: list[int]
    ) -> None:
        """Reject inputs that cannot be dispatched without losing content."""
        ...

    def fits_batch(self, *, input_count: int, token_count: int) -> bool:
        """Check inclusive hard request limits using aggregate measurements."""
        ...


class EmbeddingTransport(Protocol):
    """Dispatch one prepared batch and return same-order vectors."""

    async def embed_batch(
        self,
        request: EmbeddingRequest,
        *,
        image_uris: Mapping[tuple[str, ...], str],
    ) -> list[list[float]]:
        """Preserve native image references keyed by asset identity."""
        ...
