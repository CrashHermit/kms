"""Source triplet occurrence models."""

from pydantic import BaseModel

from kms2.core.model.base import Vertex
from kms2.core.model.source_semantic.source_entity import SourceEntity
from kms2.core.model.source_semantic.source_event import SourceEvent
from kms2.core.model.source_semantic.source_fact_extraction import SourceFact
from kms2.core.model.source_semantic.source_predicate import SourcePredicate


class SourceTriplet(Vertex):
    """Source-scoped triplet occurrence connected by typed role edges."""


class SourceTripletOccurrence(BaseModel):
    """Complete source triplet and its typed components for persistence."""

    fact: SourceFact
    triplet: SourceTriplet
    subject: SourceEntity | SourceEvent
    object: SourceEntity | SourceEvent
    predicate: SourcePredicate
