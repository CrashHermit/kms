"""Source-triplet qualifier and interpretive-binding models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class SourceTripletContextTarget(StrEnum):
    """Role within the source triplet that context qualifies or explains."""

    TRIPLET = 'triplet'
    SUBJECT = 'subject'
    PREDICATE = 'predicate'
    OBJECT = 'object'


class SourceTripletQualifierCandidate(BaseModel):
    """Source-grounded qualification attached to one triplet occurrence."""

    target: SourceTripletContextTarget
    relation: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SourceTripletInterpretiveBindingCandidate(BaseModel):
    """Source-grounded meaning attached to one triplet occurrence."""

    target: SourceTripletContextTarget
    term: str = Field(min_length=1)
    meaning: str = Field(min_length=1)


class SourceTripletQualifier(Vertex):
    """Durable qualifier owned by one source triplet occurrence."""

    source_uuid: str = Field(min_length=1)
    target: SourceTripletContextTarget
    relation: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SourceTripletInterpretiveBinding(Vertex):
    """Durable term meaning owned by one source triplet occurrence."""

    source_uuid: str = Field(min_length=1)
    target: SourceTripletContextTarget
    term: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
