"""Source-local triplet hub models."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class SourceTripletHubRole(BaseModel):
    """Canonical source hub role used in a triplet group."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceTripletHubEvidence(BaseModel):
    """One raw triplet and its source-fact evidence."""

    triplet_uuid: str = Field(min_length=1)
    fact_text: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class SourceTripletHubGroup(BaseModel):
    """One exact ordered source-local triplet-hub group."""

    subject_hub_uuid: str = Field(min_length=1)
    subject_hub: SourceTripletHubRole
    predicate_hub_uuid: str = Field(min_length=1)
    predicate_hub: SourceTripletHubRole
    object_hub_uuid: str = Field(min_length=1)
    object_hub: SourceTripletHubRole
    triplet_uuids: list[str] = Field(min_length=1)
    evidence: list[SourceTripletHubEvidence] = Field(min_length=1)


class SourceTripletHubSynthesisInput(BaseModel):
    """Fixed evidence supplied to source triplet hub synthesis."""

    subject_hub: SourceTripletHubRole
    predicate_hub: SourceTripletHubRole
    object_hub: SourceTripletHubRole
    source_facts: list[str] = Field(min_length=1)
    triplets: list[str] = Field(min_length=1)


class SourceTripletHubDefinition(BaseModel):
    """Canonical definition synthesized for one triplet group."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceTripletHub(Vertex):
    """Durable source-local triplet hub vertex."""

    source_uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    subject_hub_uuid: str = Field(min_length=1)
    predicate_hub_uuid: str = Field(min_length=1)
    object_hub_uuid: str = Field(min_length=1)


class SourceTripletHubSynthesisResult(BaseModel):
    """One ordered triplet-group synthesis result."""

    ordinal: int = Field(ge=0)
    definition: SourceTripletHubDefinition


__all__ = [
    'SourceTripletHub',
    'SourceTripletHubDefinition',
    'SourceTripletHubEvidence',
    'SourceTripletHubGroup',
    'SourceTripletHubRole',
    'SourceTripletHubSynthesisInput',
    'SourceTripletHubSynthesisResult',
]
