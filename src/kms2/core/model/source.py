"""Root source vertex model."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.base import Vertex


class SourceRevisionReference(BaseModel):
    """Concurrency token for one exact source snapshot."""

    model_config = ConfigDict(extra='forbid')

    source_uuid: str
    revision: int = Field(ge=0)


class Source(Vertex):
    """The root vertex that identifies one ingested source."""

    key: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)
