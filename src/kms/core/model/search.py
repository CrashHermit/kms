"""Typed requests for bounded relevance selection across searchable material."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from kms.core.model.content import MultimodalContent
from kms.core.model.teaching.positions import Position


class SearchMaterialKind(StrEnum):
    """Material domains accepted by the shared relevance selector."""

    SOURCE_BLOCK = 'source_block'
    STATEMENT = 'statement'
    PROCEDURE = 'procedure'
    PROPOSITION = 'proposition'
    ARGUMENT = 'argument'
    LEARNER_OBSERVATION = 'learner_observation'


class SearchSelectionCandidate(BaseModel):
    """One whole candidate in the current reranker ordering."""

    model_config = ConfigDict(extra='forbid')

    position: Position
    material: MultimodalContent
    is_exercise: bool | None = None


class SearchSelectionInput(BaseModel):
    """Query and ordered candidates supplied to the final selector."""

    model_config = ConfigDict(extra='forbid')

    query: str
    material_kind: SearchMaterialKind
    candidates: list[SearchSelectionCandidate]
