"""Ordered source units and their structural kinds."""

from enum import StrEnum

from kms.core.models import Vertex
from kms.source.models.source import SourceBlock


class SourceUnitKind(StrEnum):
    """Kinds of ordered source content."""

    CONTENT = 'content'
    EXERCISE = 'exercise'
    PROCEDURE = 'procedure'


class SourceUnit(Vertex):
    """An ordered group of source blocks."""

    kind: SourceUnitKind
    blocks: list[SourceBlock]
    embedding: list[float] | None = None
