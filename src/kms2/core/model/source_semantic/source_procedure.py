"""Persisted source procedure model."""

from pydantic import BaseModel, Field


class SourceProcedure(BaseModel):
    """Persisted reconstructed source procedure and ordered block membership."""

    uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
