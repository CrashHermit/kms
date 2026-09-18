"""Provider-neutral embedding interfaces for KMS2."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Embed ordered text inputs into one vector per input."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return vectors in the same order as the supplied texts."""
        ...


__all__ = ['EmbeddingClient']
