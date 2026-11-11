"""Typed inputs and candidate data for final search selection."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from kms.core.models import Position
from kms.media.models import Material


class SearchMaterialKind(StrEnum):
    """Material domains accepted by the shared relevance selector."""

    SOURCE_BLOCK = 'source_block'
    STATEMENT = 'statement'
    PROCEDURE = 'procedure'


class SearchSelectionCandidate(BaseModel):
    """One whole candidate in vector similarity order."""

    model_config = ConfigDict(extra='forbid')

    position: Position
    material: Material
    is_exercise: bool | None = None


class SearchSelectionInput(BaseModel):
    """Query and ordered candidates supplied to the final selector."""

    model_config = ConfigDict(extra='forbid')

    query: str
    material_kind: SearchMaterialKind
    candidates: list[SearchSelectionCandidate]
