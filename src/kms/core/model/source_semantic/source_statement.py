"""Persisted source statement model."""

from pydantic import BaseModel, Field


class SourceStatement(BaseModel):
    """Persisted reconstructed source statement and ordered block membership."""

    uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)
    is_exercise: bool = False
    content: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
