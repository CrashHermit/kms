"""Global triplet hub models."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class GlobalTripletHubRole(BaseModel):
    """Canonical global hub role used in a triplet group."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalTripletHubEvidence(BaseModel):
    """One global triplet and its source-triplet-hub evidence."""

    global_triplet_uuid: str = Field(min_length=1)
    source_triplet_hub_uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalTripletHubGroup(BaseModel):
    """One exact ordered global triplet-hub group."""

    subject_hub_uuid: str = Field(min_length=1)
    subject_hub: GlobalTripletHubRole
    predicate_hub_uuid: str = Field(min_length=1)
    predicate_hub: GlobalTripletHubRole
    object_hub_uuid: str = Field(min_length=1)
    object_hub: GlobalTripletHubRole
    global_triplet_uuids: list[str] = Field(min_length=1)
    evidence: list[GlobalTripletHubEvidence] = Field(min_length=1)


class GlobalTripletHubSynthesisInput(BaseModel):
    """Fixed evidence supplied to global triplet hub synthesis."""

    subject_hub: GlobalTripletHubRole
    predicate_hub: GlobalTripletHubRole
    object_hub: GlobalTripletHubRole
    source_triplet_hubs: list[str] = Field(min_length=1)
    global_triplets: list[str] = Field(min_length=1)


class GlobalTripletHubDefinition(BaseModel):
    """Canonical definition synthesized for one global triplet group."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalTripletHub(Vertex):
    """Durable global triplet hub vertex."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    subject_hub_uuid: str = Field(min_length=1)
    predicate_hub_uuid: str = Field(min_length=1)
    object_hub_uuid: str = Field(min_length=1)


class GlobalTripletHubSynthesisResult(BaseModel):
    """One ordered global triplet-group synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalTripletHubDefinition
