"""Source triplet occurrence models."""

from pydantic import BaseModel

from kms2.core.model.semantic.base import _SemanticOccurrence
from kms2.core.model.semantic.fact_extraction import SourceFact
from kms2.core.model.semantic.source_entity import SourceEntity
from kms2.core.model.semantic.source_event import SourceEvent
from kms2.core.model.semantic.source_predicate import SourcePredicate


class SourceTriplet(_SemanticOccurrence):
    """Source-scoped triplet occurrence connecting typed components."""

    subject_uuid: str
    object_uuid: str
    predicate_uuid: str


class SourceTripletOccurrence(BaseModel):
    """Complete source triplet and its typed components for persistence."""

    fact: SourceFact
    triplet: SourceTriplet
    subject: SourceEntity | SourceEvent
    object: SourceEntity | SourceEvent
    predicate: SourcePredicate


__all__ = ['SourceTriplet', 'SourceTripletOccurrence']
