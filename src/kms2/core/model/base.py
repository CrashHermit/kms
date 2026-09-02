"""Minimal graph record models for KMS2."""

from pydantic import BaseModel, ConfigDict


class Vertex(BaseModel):
    """One graph vertex identified by a stable UUID string."""

    model_config = ConfigDict(extra='forbid')

    uuid: str


class Edge(BaseModel):
    """One directed graph edge between source and destination vertices."""

    model_config = ConfigDict(extra='forbid')

    uuid: str
    source_uuid: str
    destination_uuid: str
