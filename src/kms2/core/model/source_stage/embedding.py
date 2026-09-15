"""Source-block embedding phase boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.page import SourcePage


class EmbeddingRequest(BaseModel):
    """Final split source pages awaiting embedding."""

    pages: list[SourcePage]


class EmbeddingResult(BaseModel):
    """Embedded source pages returned by one embedding worker."""

    pages: list[SourcePage]
