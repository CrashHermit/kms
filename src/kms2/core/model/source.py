"""Root source vertex model."""

from typing import Any

from pydantic import Field

from kms2.core.model.base import Vertex


class Source(Vertex):
    """The root vertex that identifies one ingested source."""

    key: str
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ['Source']
