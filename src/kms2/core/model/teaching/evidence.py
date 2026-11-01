"""Backend-owned references from a teaching turn to inspected evidence."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TeachingEvidenceReference(BaseModel):
    """One source block, image, or canonical proposition cited by a turn."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['block', 'image', 'proposition']
    uuid: str


class TeachingCitationView(BaseModel):
    """Learner-safe view of evidence cited in an owned teaching session."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['block', 'image', 'proposition']
    uuid: str
    content: str | None
    source_uuid: str | None = None
    source_key: str | None = None
    url: str | None = None
    title: str | None = None
    retrieved_at: datetime | None = None
    page_indexes: list[int] = Field(default_factory=list)
    image_uuids: list[str] = Field(default_factory=list)


class TeachingCitationMedia(BaseModel):
    """Authorized bytes for one image cited in an owned teaching session."""

    model_config = ConfigDict(extra='forbid')

    media_type: str
    data: bytes
