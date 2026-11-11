"""Shared graph identity types."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Vertex(BaseModel):
    """One graph vertex identified by a stable UUID string."""

    model_config = ConfigDict(extra='forbid')

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(
        default_factory=lambda data: data['created_at']
    )
    invalidated_at: datetime | None = None


class Edge(BaseModel):
    """A graph relationship with its creation time."""

    model_config = ConfigDict(extra='forbid')

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(
        default_factory=lambda data: data['created_at']
    )
    invalidated_at: datetime | None = None


Position = Annotated[int, Field(ge=0)]
