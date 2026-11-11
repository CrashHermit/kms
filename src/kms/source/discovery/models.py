"""Typed source candidates returned by discovery."""

from pydantic import BaseModel, ConfigDict


class SourceCandidate(BaseModel):
    """A text-only search candidate offered for source selection."""

    model_config = ConfigDict(extra='forbid')

    title: str | None
    url: str
    snippet: str
