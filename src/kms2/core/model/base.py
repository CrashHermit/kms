"""Minimal graph record models for KMS2."""

import uuid

from pydantic import BaseModel, ConfigDict, Field


class Vertex(BaseModel):
    """One graph vertex identified by a stable UUID string."""

    model_config = ConfigDict(extra='forbid')

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Edge(BaseModel):
    """One directed graph edge between source and destination vertices."""

    model_config = ConfigDict(extra='forbid')

    uuid: str
    source_uuid: str
    destination_uuid: str
